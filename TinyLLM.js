const text = "mama dad mama cat dad mama dog";

const words = text.split(" ");
const model = {};

for (let i = 0; i < words.length - 1; i++) {
  const a = words[i];
  const b = words[i + 1];

  model[a] ??= [];
  model[a].push(b);
}

function next(word) {
  const choices = model[word] ?? words;
  return choices[Math.floor(Math.random() * choices.length)];
}

function generate(start, n = 10) {
  let word = start;
  const out = [word];

  for (let i = 0; i < n; i++) {
    word = next(word);
    out.push(word);
  }

  return out.join(" ");
}

console.log(generate("mama"));
