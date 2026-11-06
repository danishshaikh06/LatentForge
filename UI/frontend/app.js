const promptEl = document.getElementById("prompt");
const maxTokensEl = document.getElementById("maxTokens");
const temperatureEl = document.getElementById("temperature");
const modelInfoEl = document.getElementById("modelInfo");
const statusEl = document.getElementById("status");
const responseEl = document.getElementById("response");
const generateButton = document.getElementById("generateButton");
const copyButton = document.getElementById("copyButton");

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Unable to load model info.");
    }

    if (!payload.available) {
      modelInfoEl.textContent =
        `Model status: ${payload.model} is not currently available at ${payload.base_url}.`;
      return;
    }

    modelInfoEl.textContent =
      `Model status: ${payload.model} is ready. Output mode is strict C++ code only.`;
  } catch (error) {
    modelInfoEl.textContent = `Model info unavailable: ${error.message}`;
  }
}

async function generateResponse() {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    statusEl.textContent = "Enter a prompt before generating.";
    return;
  }

  generateButton.disabled = true;
  statusEl.textContent = "Generating C++ code ...";
  responseEl.textContent = "Working...";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_new_tokens: Number(maxTokensEl.value),
        temperature: Number(temperatureEl.value),
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }

    responseEl.textContent = payload.response || "The model returned an empty response.";
    statusEl.textContent = "Generation complete.";
  } catch (error) {
    responseEl.textContent = "";
    statusEl.textContent = error.message;
  } finally {
    generateButton.disabled = false;
  }
}

async function copyResponse() {
  const text = responseEl.textContent.trim();
  if (!text || text === "Your generated C++ code will appear here.") {
    statusEl.textContent = "Generate code first, then copy it.";
    return;
  }

  await navigator.clipboard.writeText(text);
  statusEl.textContent = "Code copied to clipboard.";
}

generateButton.addEventListener("click", generateResponse);
copyButton.addEventListener("click", copyResponse);
loadModelInfo();

