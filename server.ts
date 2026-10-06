import express from "express";
import http from "http";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import { WebSocketServer, WebSocket } from "ws";
import { GoogleGenAI, Type, Modality } from "@google/genai";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      "User-Agent": "aistudio-build",
    },
  },
});

const STANCE_INSTRUCTIONS: Record<string, string> = {
  listening: `Stance: Deep Listening & Unconditional Presence.
Focus on validating the user's emotional experience without rushing to fix or advise. Mirror their feelings with warmth, name subtle emotions gently, and end with one open, tender question that invites them to share at their own pace.`,
  clarity: `Stance: Gentle Clarity & Thought Untangling.
First validate their feelings warmly. Then gently help them separate overwhelming thoughts into 2 or 3 clear, manageable threads so the weight feels lighter and less tangled. Keep tone grounded and unhurried.`,
  growth: `Stance: Courage & Fearless Growth.
Acknowledge their vulnerability with deep respect. Help them connect their current emotional challenge to their deeper values, self-compassion, and quiet personal strength. Offer one empowering perspective and a small, doable step forward.`,
  nightwatch: `Stance: Night Watch & Nervous System Soothing.
The user may be experiencing late-night anxiety, racing thoughts, or exhaustion. Use slow, rhythmic, deeply reassuring prose. Remind them that nothing needs to be solved tonight, and guide their attention toward safety and rest.`,
};

const SYSTEM_BASE = `You are HopeAI, an emotionally intelligent mental health companion designed to support, understand, and empower individuals through every emotional high and low.
Built with empathetic conversation design and a heart-first approach, you are a safe space for the user to talk freely, feel deeply, and grow fearlessly.

Core Principles:
1. Heart-First Empathy: Always acknowledge and validate feelings before offering reflection or perspective. Never sound clinical, robotic, or dismissive.
2. Nuanced Emotional Literacy: Recognize complex, layered emotions (e.g., feeling both grateful and exhausted, or hopeful yet afraid).
3. Concise, Grounded Warmth: Write in clear, breathable paragraphs (typically 2–3 short paragraphs). Avoid bullet-point overload unless untangling thoughts.
4. Safety First: If the user expresses immediate self-harm or crisis intent, respond with deep compassion and explicitly encourage them to reach out to emergency resources such as the 988 Suicide & Crisis Lifeline (call or text 988 in the US/Canada) or local emergency services immediately.`;

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "15mb" }));

  // 1. Sanctuary Companion Chat Endpoint (Supports multi-turn models: gemini-3.5-flash, gemini-3.1-pro-preview, gemini-3.1-flash-lite)
  app.post("/api/companion/chat", async (req, res) => {
    try {
      const {
        messages,
        stance = "listening",
        recentCheckIn,
        model = "gemini-3.5-flash",
        customRole = "",
      } = req.body;

      if (!Array.isArray(messages) || messages.length === 0) {
        res.status(400).json({ error: "Messages array is required." });
        return;
      }

      // Valid requested models
      const validModels = [
        "gemini-3.5-flash",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
      ];
      const selectedModel = validModels.includes(model)
        ? model
        : "gemini-3.5-flash";

      const stanceGuide =
        STANCE_INSTRUCTIONS[stance] || STANCE_INSTRUCTIONS.listening;
      const roleAugment = customRole
        ? `\nActive Specialty Role: ${customRole}\n`
        : "";
      const checkInContext = recentCheckIn
        ? `\nRecent User Emotional Check-In Context: Feeling "${recentCheckIn.primaryEmotion}" (Pleasantness: ${recentCheckIn.valence}/10, Energy: ${recentCheckIn.energy}/10), somatic sensation in ${recentCheckIn.bodyArea || "general awareness"}. Note: "${recentCheckIn.note || ""}".`
        : "";

      const conversationHistory = messages
        .map(
          (m: { role: string; content: string }) =>
            `${m.role === "user" ? "User" : "HopeAI"}: ${m.content}`
        )
        .join("\n\n");

      const prompt = `Here is the conversation so far:\n\n${conversationHistory}\n\nRespond to the user's latest message as HopeAI, and provide structured emotional resonance insights.`;

      const response = await ai.models.generateContent({
        model: selectedModel,
        contents: prompt,
        config: {
          systemInstruction: `${SYSTEM_BASE}\n\n${roleAugment}${stanceGuide}${checkInContext}`,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              reply: {
                type: Type.STRING,
                description:
                  "HopeAI's warm, empathetic response to the user (2-3 natural paragraphs).",
              },
              detectedEmotion: {
                type: Type.STRING,
                description:
                  "2-3 word nuanced description of the user's emotional state (e.g., 'Quietly overwhelmed', 'Tender & searching', 'Cautiously hopeful').",
              },
              unmetNeed: {
                type: Type.STRING,
                description:
                  "The core human need underneath the user's words (e.g., 'Permission to rest without guilt', 'Being heard without judgment', 'Reassurance & steadiness').",
              },
              gentlePrompt: {
                type: Type.STRING,
                description:
                  "A single optional follow-up reflection prompt the user can click or ponder next.",
              },
              suggestedPractice: {
                type: Type.STRING,
                description:
                  "One brief grounding or reflective micro-practice tailored to this moment (1 sentence).",
              },
            },
            required: [
              "reply",
              "detectedEmotion",
              "unmetNeed",
              "gentlePrompt",
              "suggestedPractice",
            ],
          },
        },
      });

      const rawText = response.text || "{}";
      const parsed = JSON.parse(rawText.trim());
      res.json({
        ...parsed,
        modelUsed: selectedModel,
      });
    } catch (error: any) {
      console.error("Error in /api/companion/chat:", error);
      res.status(500).json({
        error: error?.message || "Unable to reach HopeAI companion right now.",
      });
    }
  });

  // 2. Emotional Check-In Reflection Endpoint
  app.post("/api/companion/checkin-reflect", async (req, res) => {
    try {
      const {
        primaryEmotion,
        secondaryEmotion,
        valence,
        energy,
        bodyArea,
        note,
      } = req.body;

      const prompt = `The user just completed an Emotional Check-In in HopeAI:
- Primary Emotion: ${primaryEmotion}
- Secondary Nuance: ${secondaryEmotion || "None specified"}
- Pleasantness (Valence): ${valence}/10
- Energy (Arousal): ${energy}/10
- Somatic Body Area: ${bodyArea}
- Personal Note: "${note || "No additional note provided."}"

Generate a deeply empathetic, heart-first reflection that validates this exact emotional + somatic combination, explains what this state might be asking for, and offers a gentle 1-minute somatic or reflective anchor.`;

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
        config: {
          systemInstruction: SYSTEM_BASE,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              reflection: {
                type: Type.STRING,
                description:
                  "Warm 2-3 sentence compassionate reflection honoring their check-in.",
              },
              somaticAnchor: {
                type: Type.STRING,
                description:
                  "A concrete 1-sentence body-based or breath practice tailored to the body area they selected.",
              },
              affirmation: {
                type: Type.STRING,
                description:
                  "A grounded, non-cliché permission statement or affirmation for right now.",
              },
            },
            required: ["reflection", "somaticAnchor", "affirmation"],
          },
        },
      });

      const parsed = JSON.parse((response.text || "{}").trim());
      res.json(parsed);
    } catch (error: any) {
      console.error("Error in /api/companion/checkin-reflect:", error);
      res.status(500).json({
        error: error?.message || "Failed to generate check-in reflection.",
      });
    }
  });

  // 3. Cognitive Reframing Studio Endpoint ("Grow Fearlessly")
  app.post("/api/companion/reframe", async (req, res) => {
    try {
      const { thought, situation, emotionIntensity } = req.body;

      if (!thought || typeof thought !== "string") {
        res.status(400).json({ error: "A thought to reframe is required." });
        return;
      }

      const prompt = `The user shared a heavy or distressing thought to untangle and reframe:
- Thought: "${thought}"
- Situation / Context: "${situation || "General life moment"}"
- Emotional Weight: ${emotionIntensity || 7}/10

Help them feel deeply understood first, gently illuminate the mental friction pattern without clinical jargon, and offer three grounded, believable reframes along with a tiny fearless action step.`;

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
        config: {
          systemInstruction: SYSTEM_BASE,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              validation: {
                type: Type.STRING,
                description:
                  "Warm validation of why this thought makes sense given what they care about.",
              },
              patternName: {
                type: Type.STRING,
                description:
                  "Human-friendly name for the thinking pattern (e.g., 'Carrying Tomorrow's Weight Today', 'All-or-Nothing Pressure', 'Harsh Inner Critic').",
              },
              patternExplanation: {
                type: Type.STRING,
                description:
                  "1-2 sentences explaining how this thought pattern creates extra emotional friction.",
              },
              reframes: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    lens: {
                      type: Type.STRING,
                      description:
                        "Lens title: 'Self-Compassion', 'Grounded Realism', or 'Fearless Growth'.",
                    },
                    perspective: {
                      type: Type.STRING,
                      description:
                        "A believable, first-person or second-person reframed thought.",
                    },
                  },
                  required: ["lens", "perspective"],
                },
              },
              microStep: {
                type: Type.STRING,
                description:
                  "One gentle, concrete 2-minute action to move forward fearlessly.",
              },
            },
            required: [
              "validation",
              "patternName",
              "patternExplanation",
              "reframes",
              "microStep",
            ],
          },
        },
      });

      const parsed = JSON.parse((response.text || "{}").trim());
      res.json(parsed);
    } catch (error: any) {
      console.error("Error in /api/companion/reframe:", error);
      res.status(500).json({
        error: error?.message || "Failed to generate cognitive reframe.",
      });
    }
  });

  // 4. Audio Transcription Endpoint (using model gemini-3.5-transcribe)
  app.post("/api/companion/transcribe", async (req, res) => {
    try {
      const { audioBase64, mimeType = "audio/webm" } = req.body;
      if (!audioBase64) {
        res.status(400).json({ error: "Audio data is required for transcription." });
        return;
      }

      const response = await ai.models.generateContent({
        model: "gemini-3.5-transcribe",
        contents: [
          {
            inlineData: {
              data: audioBase64,
              mimeType: mimeType.split(";")[0],
            },
          },
          {
            text: "Please transcribe this microphone audio accurately and return only the transcript text. Preserve emotional pacing, pauses, and tone nuances without adding conversational replies.",
          },
        ],
      });

      const transcription = response.text || "";
      res.json({ transcription: transcription.trim() });
    } catch (error: any) {
      console.error("Error in /api/companion/transcribe:", error);
      res.status(500).json({
        error:
          error?.message ||
          "Unable to transcribe audio with gemini-3.5-transcribe.",
      });
    }
  });

  // 5. Music Generation Endpoint (using lyria-3-clip-preview and lyria-3-pro-preview)
  app.post("/api/companion/generate-music", async (req, res) => {
    try {
      const {
        prompt,
        model = "lyria-3-clip-preview",
        mood = "calming",
      } = req.body;

      if (!prompt || typeof prompt !== "string") {
        res.status(400).json({ error: "Music prompt is required." });
        return;
      }

      const selectedModel =
        model === "lyria-3-pro-preview"
          ? "lyria-3-pro-preview"
          : "lyria-3-clip-preview";

      const enrichedPrompt = `Create a meditative, emotionally restorative ambient soundtrack. Style: ${prompt}. Mood: ${mood}. Warm analog textures, binaural harmonics, gentle soothing tempo.`;

      const responseStream = await ai.models.generateContentStream({
        model: selectedModel,
        contents: enrichedPrompt,
        config: {
          responseModalities: [Modality.AUDIO],
        },
      });

      let accumulatedAudioBase64 = "";
      let metadataOrLyrics = "";
      let resolvedMimeType = "audio/wav";

      for await (const chunk of responseStream) {
        const parts = chunk.candidates?.[0]?.content?.parts;
        if (!parts) continue;

        for (const part of parts) {
          if (part.inlineData?.data) {
            if (!accumulatedAudioBase64 && part.inlineData.mimeType) {
              resolvedMimeType = part.inlineData.mimeType;
            }
            accumulatedAudioBase64 += part.inlineData.data;
          }
          if (part.text && !metadataOrLyrics) {
            metadataOrLyrics = part.text;
          }
        }
      }

      if (!accumulatedAudioBase64) {
        throw new Error(
          "Lyria stream completed without audio data. Please ensure your GEMINI_API_KEY supports Lyria music generation."
        );
      }

      res.json({
        audioBase64: accumulatedAudioBase64,
        mimeType: resolvedMimeType,
        metadata: metadataOrLyrics,
        modelUsed: selectedModel,
      });
    } catch (error: any) {
      console.error("Error in /api/companion/generate-music:", error);
      res.status(500).json({
        error:
          error?.message ||
          "Music generation with Lyria is unavailable or requires a paid API key tier.",
      });
    }
  });

  // 6. Calming Voice Read-Aloud Endpoint (TTS)
  app.post("/api/companion/speak", async (req, res) => {
    try {
      const { text } = req.body;
      if (!text || typeof text !== "string") {
        res.status(400).json({ error: "Text is required for speech synthesis." });
        return;
      }

      const trimmedText = text.slice(0, 900);

      const response = await ai.models.generateContent({
        model: "gemini-3.8-flash-lite-tts",
        contents: [
          {
            role: "user",
            parts: [
              {
                text: trimmedText,
                speechMetadata: {
                  style:
                    "Warm, calm, empathetic, unhurried mental health companion",
                },
              } as any,
            ],
          },
        ],
        config: {
          responseModalities: ["AUDIO"],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: "Kore" },
            },
          },
        },
      });

      const part = response.candidates?.[0]?.content?.parts?.[0];
      const base64Audio = part?.inlineData?.data;
      const mimeType = part?.inlineData?.mimeType || "audio/pcm;rate=24000";

      if (!base64Audio) {
        res.status(500).json({ error: "No audio returned from speech model." });
        return;
      }

      res.json({ audioBase64: base64Audio, mimeType, sampleRate: 24000 });
    } catch (error: any) {
      console.error("Error in /api/companion/speak:", error);
      res.status(500).json({
        error: error?.message || "Unable to synthesize calming voice right now.",
      });
    }
  });

  // 7. Personalized Daily Intention Suggestion Endpoint
  app.post("/api/companion/intention-suggest", async (req, res) => {
    try {
      const { currentMood, focusArea } = req.body;
      const prompt = `The user is setting a simple, heart-first Daily Intention in HopeAI to build gentle positive momentum.
Current emotional state or check-in: "${currentMood || "Seeking steadiness and gentle progress"}"
Desired area of focus: "${focusArea || "Mindful Presence"}"

Suggest one small, deeply doable, emotionally intelligent daily intention (not a high-pressure productivity chore) and explain in one warm sentence why honoring this small step builds self-trust.`;

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
        config: {
          systemInstruction: SYSTEM_BASE,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: {
                type: Type.STRING,
                description:
                  "A concise, concrete, compassionate daily intention (under 14 words).",
              },
              category: {
                type: Type.STRING,
                description:
                  "One of: 'Rest & Boundaries', 'Gentle Connection', 'Mindful Presence', or 'Fearless Step'.",
              },
              whyItMatters: {
                type: Type.STRING,
                description:
                  "1 warm sentence on how this intention supports emotional wellbeing and momentum.",
              },
            },
            required: ["title", "category", "whyItMatters"],
          },
        },
      });

      const parsed = JSON.parse((response.text || "{}").trim());
      res.json(parsed);
    } catch (error: any) {
      console.error("Error in /api/companion/intention-suggest:", error);
      res.status(500).json({
        error:
          error?.message ||
          "Unable to craft a personalized intention right now.",
      });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  const PORT = 3000;
  const server = http.createServer(app);

  // Setup WebSocket Server for Live API voice conversations (gemini-3.8-live)
  const wss = new WebSocketServer({ server, path: "/ws/live" });

  wss.on("connection", async (clientWs: WebSocket) => {
    console.log("Client connected to HopeAI Live Voice Sanctuary");
    let session: any = null;

    try {
      session = await ai.live.connect({
        model: "gemini-3.8-live",
        callbacks: {
          onmessage: (message: any) => {
            const audio =
              message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audio && clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ audio }));
            }
            if (
              message.serverContent?.interrupted &&
              clientWs.readyState === WebSocket.OPEN
            ) {
              clientWs.send(JSON.stringify({ interrupted: true }));
            }
          },
          onclose: () => {
            if (clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ status: "closed" }));
            }
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
          systemInstruction: `${SYSTEM_BASE}\n\nYou are in a live, real-time voice conversation with the user. Speak with warm, unhurried cadence, gentle voice tones, and grounded reassurance. Keep your voice turns natural and brief (1-3 sentences) so the dialogue feels like a steady companion sitting right beside them.`,
        },
      });

      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(JSON.stringify({ status: "connected" }));
      }
    } catch (err: any) {
      console.error("Error connecting to Gemini Live API:", err);
      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(
          JSON.stringify({
            error:
              err?.message ||
              "Could not establish Live API voice session. Please ensure your key supports real-time audio.",
          })
        );
      }
    }

    clientWs.on("message", (data: any) => {
      try {
        const payload = JSON.parse(data.toString());
        if (payload.audio && session) {
          session.sendRealtimeInput({
            audio: {
              data: payload.audio,
              mimeType: "audio/pcm;rate=16000",
            },
          });
        }
      } catch (err) {
        console.warn("Invalid live client message:", err);
      }
    });

    clientWs.on("close", () => {
      console.log("Client disconnected from Live Sanctuary");
      if (session && typeof session.close === "function") {
        try {
          session.close();
        } catch {}
      }
    });
  });

  server.listen(PORT, "0.0.0.0", () => {
    console.log(`HopeAI server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
