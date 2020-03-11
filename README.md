# Novoic-Audio-Challenge
This will be where files, plots and scripts created throughout this project will be stored.

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

