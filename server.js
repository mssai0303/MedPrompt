

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();


const PORT  = process.env.PORT  || 8787;
const MODEL = process.env.MODEL || "gemini-1.5-flash";
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

if (!GOOGLE_API_KEY) {
  console.error("❌ Missing GOOGLE_API_KEY in .env");
  process.exit(1);
}

// --- Middleware ---
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(express.static("public")); // serves public/index.html

// --- Gemini client ---
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);

// --- System instruction / guardrails ---
const SYSTEM_PROMPT = `You are a careful clinical reasoning assistant.
- You are NOT a doctor and cannot provide medical diagnosis or treatment.
- Always include: (1) brief summary; (2) differential diagnoses with rationale; (3) home care steps; (4) clear red flags; (5) short disclaimer.
- Keep answers under 350 words unless asked.
- Use cautious language ("might", "could") and avoid definitive treatment plans.
- Refuse prescriptions, image test-result interpretation, or personalized medical orders.
Return ONLY JSON with keys:
  summary (string),
  differential (array of {condition, why}),
  homeCare (array of string),
  redFlags (array of string),
  disclaimer (string).`;

// Helper: call Gemini and force JSON output
async function analyzeWithGemini(userText) {
  const model = genAI.getGenerativeModel({
    model: MODEL,
    systemInstruction: SYSTEM_PROMPT,
  });

  // Force JSON back
  const result = await model.generateContent({
    contents: [
      {
        role: "user",
        parts: [{ text: `Patient message (anonymous): ${userText}` }],
      },
    ],
    generationConfig: {
      responseMimeType: "application/json",
      temperature: 0.4,
      maxOutputTokens: 700,
    },
  });

  // Gemini returns a text() string (should be valid JSON because of responseMimeType)
  return result.response.text();
}

// --- API route ---
app.post("/analyze", async (req, res) => {
  try {
    const { userText, mock } = req.body || {};

    // Optional mock mode for debugging
    if (mock === true || process.env.MOCK === "1") {
      return res.json({
        ok: true,
        data: {
          summary: `Echo: ${userText}`,
          differential: [{ condition: "Common cold", why: "Mock mode" }],
          homeCare: ["Rest", "Fluids"],
          redFlags: ["High fever > 39.4°C"],
          disclaimer: "Demo only.",
        },
      });
    }

    if (!userText || typeof userText !== "string") {
      return res
        .status(400)
        .json({ ok: false, error: "userText (string) is required" });
    }

    const raw = await analyzeWithGemini(userText);

    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      // If the model ever returns non-JSON, still show something
      data = { summary: raw };
    }

    const normalized = {
      summary: data.summary || "",
      differential: Array.isArray(data.differential) ? data.differential : [],
      homeCare: Array.isArray(data.homeCare) ? data.homeCare : [],
      redFlags: Array.isArray(data.redFlags) ? data.redFlags : [],
      disclaimer:
        data.disclaimer ||
        "Educational use only. Not a diagnosis or treatment. See a licensed clinician for care.",
    };

    res.json({ ok: true, data: normalized });
  } catch (err) {
    const status = err?.status || err?.response?.status;
    const detail = err?.response?.data || err?.message || String(err);
    console.error("Analyze error:", status, detail);

    // Friendly message for common cases
    let userMessage = "AI request failed. Try again.";
    if (status === 401 || /API key/i.test(detail)) userMessage = "Invalid Google API key. Update GOOGLE_API_KEY and restart.";
    if (status === 429 || /quota|rate limit/i.test(detail)) userMessage = "Quota exceeded. Check your Google AI Studio billing/limits.";
    if (/model/i.test(detail) && /not|unknown/i.test(detail)) userMessage = "Model not available. Try MODEL=gemini-1.5-flash.";

    return res.status(500).json({
      ok: false,
      error: userMessage,
      status,
    });
  }
});

// --- Start ---
app.listen(PORT, () => {
  console.log(`MedPrompt backend (Gemini) listening on http://localhost:${PORT}`);
});
