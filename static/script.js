const form = document.getElementById("form");
const btn = document.getElementById("btn");
const errorBox = document.getElementById("error");
const result = document.getElementById("result");
const bar = document.getElementById("bar");
const OPTIONAL = ["Cholesterol", "RestingBP"];

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.textContent = "";
  const data = {};
  let missing = false;

  for (const el of form.elements) {
    if (!el.name) continue;
    el.classList.remove("invalid");
    if (el.value === "" && !OPTIONAL.includes(el.name)) {
      el.classList.add("invalid");
      missing = true;
    }
    data[el.name] = el.value;
  }
  if (missing) {
    errorBox.textContent = "Please fill in the highlighted fields.";
    return;
  }

  btn.disabled = true;
  btn.textContent = "Analysing…";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || "Something went wrong.");
    showResult(out.probability);
  } catch (err) {
    errorBox.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict risk";
  }
});

function showResult(p) {
  let color, title;
  if (p < 30) { color = "var(--ok)"; title = "Low risk"; }
  else if (p < 60) { color = "var(--warn)"; title = "Moderate risk"; }
  else { color = "var(--accent)"; title = "High risk"; }

  document.getElementById("label").textContent = `${title} (${p}%)`;
  document.getElementById("label").style.color = color;
  document.getElementById("detail").textContent =
    p >= 40
      ? "The model sees patterns similar to patients with heart disease. Please consult a cardiologist."
      : "The model sees patterns similar to patients without heart disease. Keep up regular check-ups.";
  bar.style.background = color;
  bar.style.width = "0";
  result.hidden = false;
  requestAnimationFrame(() => (bar.style.width = p + "%"));
  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.getElementById("reset").addEventListener("click", () => {
  result.hidden = true;
  errorBox.textContent = "";
  form.querySelectorAll(".invalid").forEach((el) => el.classList.remove("invalid"));
});
