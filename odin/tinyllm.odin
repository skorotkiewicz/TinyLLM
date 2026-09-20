// TinyLLM in Odin — bigram Markov chain, port of your JS version
// run: odin run . (from this directory, file: tinyllm.odin)
package main

import "core:fmt"
import "core:math/rand"
import "core:strings"

main :: proc() {
	text := "mama dad mama cat dad mama dog"
	words := strings.fields(text)

	// train: word -> [next words]
	model := make(map[string][]string)
	for i in 0 ..< len(words) - 1 {
		model[words[i]] = append(model[words[i]], words[i + 1])
	}

	// generate: random walk
	word := words[0]
	fmt.print(word)
	for _ in 0 ..< 10 {
		if choices, ok := model[word]; ok {
			word = choices[rand.uint64() % uint64(len(choices))]
		} else {
			word = words[rand.uint64() % uint64(len(words))]
		}
		fmt.print(" {}", word)
	}
	fmt.println()
}
