

# Machine Made Media
GitHub repository for websites utilized to document the change in the rates of machine-generated/synthetic articles between January 1, 2022, and May 1, 2023.

We collected articles from these websites between January 1, 2022, and May 1, 2023. 

This dataset consists of 1,059 misinformation/unreliable news websites and 2,015 reliable/mainstream news websites. 

For additional details about the collection method and analysis of these websites' connection with other misinformation-related websites as well as more reliable mainstream news websites, see our paper/analysis here: [https://arxiv.org/pdf/2301.10880.pdf](https://arxiv.org/pdf/2305.09820.pdf). 

Our code for perturbing news articles is adapted from https://github.com/eric-mitchell/detect-gpt/blob/main/run.py and our code for paraphrasing articles is adapted from https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl

## Request Machine Made Media Website URLs
Over the course of the period of study for this work, we collected the published articles from our list of websites' RSS feeds and from querying each website's homepage. Please fill out the following [Google form](https://forms.gle/vGhZKkG5jy1cT7dx6) for access to an extended set of Article URLs from the websites used in this study. This dataset may only be utilized for research purposes, the copyright of the articles within this dataset belongs to the respective websites. 

## Request DeBERTA Model Weights
In this work, we utilized a finetuned version of the DeBERTA-v3-base model to differentiate between synthetic and human-written news articles. To request the weights of the model used in this work, please fill out the following [Google form](https://forms.gle/4WeVk8FwTafhtiU78)

## Citing the paper
If our lists of websites, the results from our paper, or our URLs are useful for your own research, you can cite us with the following BibTex entry:
```
  @inproceedings{hanley2024machine,
    title={Machine-Made Media: Monitoring the Mobilization of Machine-Generated Articles on Misinformation and Mainstream News Websites},
    author={Hanley, Hans WA and Durumeric, Zakir},
    booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
    volume={18},
    year={2024}
  }
```

## License and Copyright

Copyright 2024 The Board of Trustees of The Leland Stanford Junior University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
