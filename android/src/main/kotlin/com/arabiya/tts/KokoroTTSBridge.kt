package com.arabiya.tts

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import org.json.JSONObject
import java.nio.FloatBuffer

/**
 * Kokoro TTS Bridge — ONNX Runtime on Android
 * 
 * Converts prosody-annotated Arabic text into PCM audio using
 * the Kokoro 82M ONNX model.
 * 
 * Pipeline:
 *   ProsodyPlan (JSON) → Phonemize → Kokoro ONNX → PCM → AudioTrack
 * 
 * Usage:
 *   val tts = KokoroTTSBridge(context)
 *   tts.speak(prosodyJson)  // Non-blocking, queues audio
 *   tts.release()           // Cleanup
 */
class KokoroTTSBridge(
    private val context: Context,
    private val modelPath: String = "kokoro_v0_19.onnx",
    private val voicesPath: String = "voices.bin",
) {
    // ONNX Runtime
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    
    // Audio playback
    private var audioTrack: AudioTrack? = null
    private val sampleRate = 24000
    
    // Voice style (default: Arabic female)
    private var voiceData: FloatArray? = null
    private var voiceStyleIndex = 0
    
    /**
     * Initialize the Kokoro ONNX model and audio track.
     */
    fun initialize() {
        // Load ONNX model from assets
        val modelBytes = context.assets.open(modelPath).readBytes()
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
        }
        session = env.createSession(modelBytes, sessionOptions)
        
        // Load voice embeddings
        context.assets.open(voicesPath).use { stream ->
            val bytes = stream.readBytes()
            voiceData = FloatArray(bytes.size / 4).also { arr ->
                java.nio.ByteBuffer.wrap(bytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .get(arr)
            }
        }
        
        // Initialize AudioTrack for low-latency playback
        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT
        )
        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize * 2)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        
        audioTrack?.play()
    }
    
    /**
     * Speak a prosody plan (from the Arabiya TTS pipeline).
     * 
     * @param prosodyJson JSON string from ArabiyaTTSPipeline.toJson()
     */
    fun speak(prosodyJson: String) {
        val plan = JSONObject(prosodyJson)
        val segments = plan.getJSONArray("segments")
        val style = plan.getJSONObject("style")
        
        val globalSpeed = style.optDouble("speed", 1.0).toFloat()
        val globalPitch = style.optDouble("pitch", 1.0).toFloat()
        
        // Process each segment
        for (i in 0 until segments.length()) {
            val seg = segments.getJSONObject(i)
            val word = seg.getString("word")
            val speedFactor = seg.optDouble("speed_factor", 1.0).toFloat()
            val pauseMs = seg.optInt("pause_after_ms", 0)
            
            // Synthesize word audio
            val audio = synthesizeWord(word, globalSpeed * speedFactor, globalPitch)
            if (audio != null) {
                audioTrack?.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
            }
            
            // Insert pause
            if (pauseMs > 0) {
                val silenceSamples = (sampleRate * pauseMs / 1000.0).toInt()
                val silence = FloatArray(silenceSamples)
                audioTrack?.write(silence, 0, silence.size, AudioTrack.WRITE_BLOCKING)
            }
        }
    }
    
    /**
     * Synthesize a single word/phrase using Kokoro ONNX.
     */
    private fun synthesizeWord(text: String, speed: Float, pitch: Float): FloatArray? {
        val sess = session ?: return null
        
        // Phonemize Arabic text → token IDs
        val tokenIds = phonemize(text)
        if (tokenIds.isEmpty()) return null
        
        // Build ONNX inputs
        val inputIds = LongArray(tokenIds.size) { tokenIds[it].toLong() }
        val inputShape = longArrayOf(1, inputIds.size.toLong())
        
        val inputTensor = OnnxTensor.createTensor(env, 
            java.nio.LongBuffer.wrap(inputIds), inputShape)
        
        // Style embedding (256-dim voice vector)
        val styleShape = longArrayOf(1, 256)
        val styleEmbed = getVoiceStyle(voiceStyleIndex)
        val styleTensor = OnnxTensor.createTensor(env,
            FloatBuffer.wrap(styleEmbed), styleShape)
        
        // Speed tensor
        val speedTensor = OnnxTensor.createTensor(env,
            FloatBuffer.wrap(floatArrayOf(speed)), longArrayOf(1))
        
        val inputs = mapOf(
            "input_ids" to inputTensor,
            "style" to styleTensor,
            "speed" to speedTensor,
        )
        
        return try {
            val results = sess.run(inputs)
            val audioTensor = results[0] as OnnxTensor
            val audio = audioTensor.floatBuffer.let { buf ->
                FloatArray(buf.remaining()).also { buf.get(it) }
            }
            results.close()
            audio
        } catch (e: Exception) {
            null
        } finally {
            inputTensor.close()
            styleTensor.close()
            speedTensor.close()
        }
    }
    
    /**
     * Arabic phonemizer — converts Arabic text to Kokoro token IDs.
     * 
     * Maps Arabic characters to IPA-like phonemes that Kokoro understands.
     */
    private fun phonemize(text: String): IntArray {
        // Arabic letter → phoneme mapping
        val phonemeMap = mapOf(
            'ا' to "aː", 'ب' to "b", 'ت' to "t", 'ث' to "θ",
            'ج' to "dʒ", 'ح' to "ħ", 'خ' to "x", 'د' to "d",
            'ذ' to "ð", 'ر' to "r", 'ز' to "z", 'س' to "s",
            'ش' to "ʃ", 'ص' to "sˤ", 'ض' to "dˤ", 'ط' to "tˤ",
            'ظ' to "ðˤ", 'ع' to "ʕ", 'غ' to "ɣ", 'ف' to "f",
            'ق' to "q", 'ك' to "k", 'ل' to "l", 'م' to "m",
            'ن' to "n", 'ه' to "h", 'و' to "w", 'ي' to "j",
            'ء' to "ʔ", 'ة' to "a", 'ى' to "aː",
            'أ' to "ʔa", 'إ' to "ʔi", 'آ' to "ʔaː", 'ؤ' to "ʔu",
            'ئ' to "ʔi",
            // Diacritics → vowels
            '\u064E' to "a", '\u064F' to "u", '\u0650' to "i",
            '\u064B' to "an", '\u064C' to "un", '\u064D' to "in",
            '\u0652' to "", '\u0651' to "", // sukun, shadda
        )
        
        // Convert to phoneme sequence
        val phonemes = StringBuilder()
        var prevWasShadda = false
        
        for (c in text) {
            if (c == '\u0651') { // shadda = gemination
                prevWasShadda = true
                continue
            }
            val p = phonemeMap[c]
            if (p != null) {
                if (prevWasShadda && p.isNotEmpty()) {
                    phonemes.append(p) // double the consonant
                }
                phonemes.append(p)
                prevWasShadda = false
            } else if (c == ' ') {
                phonemes.append(' ')
                prevWasShadda = false
            }
        }
        
        // Map phoneme string to token IDs (simplified — real impl uses vocab)
        return phonemes.toString().map { it.code }.toIntArray()
    }
    
    /**
     * Get voice style embedding for given index.
     */
    private fun getVoiceStyle(index: Int): FloatArray {
        val voice = voiceData ?: return FloatArray(256)
        val offset = index * 256
        return if (offset + 256 <= voice.size) {
            voice.copyOfRange(offset, offset + 256)
        } else {
            FloatArray(256)
        }
    }
    
    /**
     * Set voice style by name.
     */
    fun setVoice(name: String) {
        voiceStyleIndex = when (name.lowercase()) {
            "af_heart" -> 0   // Arabic female
            "af_star" -> 1    // Arabic female alt
            "am_adam" -> 2    // Arabic male
            else -> 0
        }
    }
    
    /**
     * Release all resources.
     */
    fun release() {
        audioTrack?.apply {
            stop()
            release()
        }
        audioTrack = null
        session?.close()
        session = null
    }
}
