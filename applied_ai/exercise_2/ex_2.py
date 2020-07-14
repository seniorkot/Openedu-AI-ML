import speech_recognition as sr


def print_results(audios: list):
    r = sr.Recognizer()

    for audio_id in audios:
        audio_name = 'exercise_2/audio/' + str(audio_id) + '.wav'
        with sr.AudioFile(audio_name) as source:
            audio = r.record(source)

        print('\n' + str(audio_id) + '.wav:')

        # Sphinx
        try:
            print('Sphinx: ' + r.recognize_sphinx(audio))
        except sr.UnknownValueError:
            print('Sphinx couldn\'t understand audio')
        except sr.RequestError as e:
            print('Sphinx error; {0}'.format(e))

        # Google
        try:
            print('Google: ' + r.recognize_google(audio))
        except sr.UnknownValueError:
            print('Google Speech Recognition could not understand audio')
        except sr.RequestError as e:
            print('Google error; {0}'.format(e))
