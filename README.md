# MGT Mark

A high(ish) performance CUDA accelerated MGT (Machine Generated Text) obfuscation and detection benchmarking tool Supports HuggingFace classification pipelines as well as commercial APIs (beta). Datasets and models can be pulled straight from HuggingFace repos, and then applied with attacks. 

Should give a strong indication on what detection systems are affective against, and whether they can be defeated through MGT obfuscation. Perpetual understatement aside, this is actually quite important. LLMs are here and there here to stay, and we need a way of easily testing detection systems. Importantly, we also need a way of testing the affects of mgt obfuscation techniques. 



## Supported Attacks

The following attacks are implemented:
1. Homoglyph attacks - swap characters around to break up tokenisation e.g. LATIN "O" -> Greek "O". Can be configured to swap any characters so go wild with it. Can increase entropy significantly so would also affect any system that uses perplexity.  
2. ZWSP padding - insert zero width spaces to pad text, again this breaks up tokenisation. Can reduce entropy to almost 0. Be careful with this one, if testing commercial APIs cause it might break stuff. You can also produce an arbitrarily long entry, so it might get a little expensive. 
3. Spelling - swap whole words, software comes packaged with a basic list of commonly misspelled words, but again you can make your own. Breaks up tokenisation, but does not significantly affect entropy . 
4. Strategic Spaces - Based of the implementation by X, adds spaces before commas to break up tokenisation.
5. Paraphrasing - Use GT3.5 api to rewrite entries, helps to move text from the training data of detection systems.


Data shows (Report will be added here in future) that all these attacks are very affective and can cripple the performance of open-source classifiers and commercial tools, even when minimal options are used. 


## Configuration

Can be run with the command line (see help screen), if only one dataset or model is to be used. More likely you'll want to run a whole bunch of stuff all at once, a JSON config file can be used to do so, first add the datasets i.e

```json
{
  "dataset": [
    {
      "name": "aadityaubhat/GPT-wiki-intro",
      "human_samples": "wiki_intro",
      "machine_samples": "generated_intro"
    }
  ]
}
```
Then add the attacks:

```json
  "attacks": [
    {
      "name": "glyph",
      "pair_file": "config/subtle_pairs.json",
      "chance": 0.25
    },
    {
      "name": "zwsp",
      "padding_mult": 0.01
    },
    {
      "name": "spelling",
      "spell_file": "config/spelling.dat",
      "chance": 0.25
    },
    {
      "name": "spacing",
      "chance": 0.25
    }
  ],
```

Finally add the detection systems:
```json
"models": [
    {
      "name": "andreas122001/roberta-wiki-detector",
      "human_label": "human-produced",
      "machine_label": "machine-generated"
    },
    {
      "name": "ORIG",
      "auth_file": "config/orig_auth.key",
      "human_label": "Real",
      "machine_label": "Fake"
    }
]
```

Apply the config file via the **-c** flag and limit the maximum number of samples with **-s** or run all samples by leaving it. 

## Contributing

If anyone's bored enough to actually contribute to this then do so. 

## Licence
