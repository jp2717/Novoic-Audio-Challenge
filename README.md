# Novoic-Audio-Challenge
This will be where files, plots and scripts created throughout this project will be stored.

'functions.py' => where functions are stored
'feature_extraction.py' => where the features desired are extracted from audio files
'learning.py' => where a traditional machine learning algorithm is used to analyse and classify data

For this project, I chose to focus on data exploration/visualization as well as binary and multiclass classfication. After in-depth research on audio signal processing, I chose to use a typical machine learning approach after engineering relevant features. As all words are monosyllabic (some with monophthongs and some with diphthongs), three key features are the first three formants of the audio signal, which were computed via a personalised function that implements linear predictive coding. As there are differences in average pitches between various people, the mean of the three formants for each sample was also recorded, almost as a reference point. Similarly, the difference between the first two formants is a common classfier of vowels. To account for irregularities, the duration of the sound (not the same as the audio file duration) was taken as another feature, which may also shine some light into words with nasal sounds. As a rough classifier for plosives, the difference between the peak and mean intensity was taken and normalised by the peak(to account for different amplitudes) and multipied by the duration of the sound(s), to account for the mean of a signal depending on its duration (essentially drops to 0 when the word is not being spoken). 

Features:
1. F1(Hz)
2. F2(Hz)
3. F3(Hz)
4. (F1 + F2 + F3)/3 (Hz)
5. F2 - F1 (Hz)
6. Duration(ms)
7. Relative Amplitude

The machine learning algorithm is a multiclass, one-vs-rest approach, implementing Linear SVMs, which utilises the seven features engineered to classify audio files.

Results (using Linear SVMs):

Multiclass Linear Classification:
Training accuracy: 24.5%
Test accuracy: 24.4%

Simple Binary Classification(all combinations of two words from the 10 provided):

Top 3 Combos(where algorithm had most success)

'off' and 'yes': (seemed to be most differentiable for F2vsF3 and F2vs(F2-F1))
Training accuracy: 83.0%
Test accuracy: 81.7%

'no' and 'up':    (Relative amplitude seemed to be the most differentiable out of the features, perhaps due to the plosive consonant, 'p')
Training accuracy: 83.2%
Test accuracy: 82.0%

'no' and 'off':
Training accuracy: 80.8%
Test accuracy: 79.7%

Mean accuracy (of all combinations):
Training accuracy: 62.0%
Test accuracy: 61.3.0%

Note:

The worst classification scores occured for 'go' and 'left', 'go' and 'right','left' and 'stop'. This is perhaps because the feature used to distinguish between plosives does not distinguish between them. The same could be argued for the success of distinguishing between 'no' and 'up'.

