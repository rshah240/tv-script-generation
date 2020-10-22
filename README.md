# tv-script-generation
Word Level LSTM model to Generate tv scripts

## Usage
To train the Model
```bash
python tv-script-generator.py --data='./data/Friends_Script.txt'
```
To generate the text
```bash
python generate.py --prime='Joey' --model='Model_Friends.pt' --length=400 -data='./data/Friends_Script.txt'
```

## To download the Trained Model
I have trained model for two tv-scripts:- ie Friends and Seinfeld. 

[Click here to download Friends Model](https://drive.google.com/file/d/1WlCHaqzvFP3RmWg-TWMXPA7T3FWxrF8v/view?usp=sharing)
[Click here to download Seinfeld Model](https://drive.google.com/file/d/1F0egWRXA3BWGWRaa2T5t5fRJqqEztviz/view?usp=sharing)
