const VAL_PATH = "data/val.jsonl";
const API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"; // OpenAI compatible endpoint :contentReference[oaicite:3]{index=3}

let valExamples = [];

async function loadValidationData() {
  const res = await fetch(VAL_PATH);
  const text = await res.text();
  valExamples = text
    .split("\n")
    .filter(line => line.trim().length > 0)
    .map(line => JSON.parse(line));

  const select = document.getElementById("example-select");
  valExamples.forEach((ex, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `Example ${ex.id}`;
    select.appendChild(opt);
  });

  updateExampleDisplay(0);
}

function updateExampleDisplay(index) {
  const ex = valExamples[index];
  const inputBox = document.getElementById("example-input");
  const outputBox = document.getElementById("example-output");

  inputBox.textContent = `Instruction:\n${ex.instruction}\n\nInput:\n${ex.input}`;
  outputBox.textContent = ex.output;

  document.getElementById("base-answer").textContent = "";
  document.getElementById("ft-answer").textContent = "";
  document.getElementById("metrics-text").textContent = "Run an example to see scores.";
}

function buildMessages(ex) {
  return [
    {
      role: "user",
      content: `Instruction: ${ex.instruction}\nInput: ${ex.input}\nReply:`
    }
  ];
}

async function callDeepInfra(apiKey, modelName, messages) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: modelName,
      messages,
      max_tokens: 256,
      temperature: 0.3
    })
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }

  const data = await res.json();
  const choice = data.choices && data.choices[0];
  if (!choice) return "";
  return choice.message.content || "";
}

function computeSimpleScore(pred, gold) {
  const norm = s => s.toLowerCase().replace(/\s+/g, " ").trim();
  const p = norm(pred);
  const g = norm(gold);

  if (!p.length || !g.length) return 0;

  if (p === g) return 1;

  const pTokens = new Set(p.split(" "));
  const gTokens = g.split(" ");
  let overlap = 0;
  gTokens.forEach(t => {
    if (pTokens.has(t)) overlap += 1;
  });

  return overlap / gTokens.length;
}

document.addEventListener("DOMContentLoaded", () => {
  loadValidationData();

  const select = document.getElementById("example-select");
  const runBtn = document.getElementById("run-btn");

  select.addEventListener("change", e => {
    updateExampleDisplay(Number(e.target.value));
  });

  runBtn.addEventListener("click", async () => {
    const apiKey = document.getElementById("api-key-input").value.trim();
    const baseModel = document.getElementById("base-model-input").value.trim();
    const ftModel = document.getElementById("ft-model-input").value.trim();

    if (!apiKey) {
      alert("Please paste your DeepInfra API key first.");
      return;
    }

    const idx = Number(select.value);
    const ex = valExamples[idx];
    const messages = buildMessages(ex);

    runBtn.disabled = true;
    runBtn.textContent = "Running...";

    try {
      const [baseAns, ftAns] = await Promise.all([
        callDeepInfra(apiKey, baseModel, messages),
        callDeepInfra(apiKey, ftModel, messages)
      ]);

      document.getElementById("base-answer").textContent = baseAns;
      document.getElementById("ft-answer").textContent = ftAns;

      const baseScore = computeSimpleScore(baseAns, ex.output);
      const ftScore = computeSimpleScore(ftAns, ex.output);

      document.getElementById("metrics-text").textContent =
        `Token overlap score (0–1): base = ${baseScore.toFixed(2)}  fine-tuned = ${ftScore.toFixed(2)}.`;
    } catch (err) {
      document.getElementById("metrics-text").textContent =
        "Error calling API: " + err.message;
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "Run models";
    }
  });
});
