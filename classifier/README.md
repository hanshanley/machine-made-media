# Code for training DeBERTa-v3 classifier

This code requires transformers==4.30.2 to run. This can be installed by running
```
pip3 install transformers==4.30.2
```

If you train a model using a later version of the transformers library but still wish to use the weights utilized in our paper, please pass the following argument ```strict=False``` when loading the model weights with the ```load_state_dict``` method.
