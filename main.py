from flask import Flask, render_template,request
import camera
import os
from base64 import b64encode
from gtts import gTTS



start_data = 'data:audio/mpeg;base64,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cC0YsJprsrzBF3xjOMY/g6pX+rLRc3A8mKajE+juZlB0VMbM6kEHapAou2cJb/2ontzP+/+igRwExEpxyMPnbhoWdNmDllpU+beUxVu+9TPQhQAB//NExGQRcSJ8AOvacNBKTLjD0hh1LSgy05U3NM/0zy2+Fz6d/f//6+fz8If0gyc7u/uro2/TcM+zkxsPP3NuCYYFCCCaxIQOnNd6hAYXRoEH/rNVDEekSCMGwoZYmkqT//NExHEguxqIANjSvallHQxIUIHaomABCjlbaZO2HzFVrtdC3KdWuGb5iT11uuismrBinGRzs8+vdule3PvWu09MzM9eZnM/t6/5MzMzNLbMNpbztY+aT16OfezMc1tC//NExEEgkxacAHiYufr19mlsT2ajUqoCYjTchkQzbcuqSvrikYpV7iEnJB6ekklMIyKDhmTSaeXHsf1R60y2ycVOYvMBFfU+WlB4XbBO1Rm7lLp///////7//////in3//NExBETMxK4ABAWuT/H/F/+10vlzKmbk3YmscOFBk1U+40NDQ/CxPckyIM0kEDd1DscsbOMVDlKjv1TeXHTdFjVzo3jvE5uXQVf////////////L////9X/k6rZ+w5H//NExBcSQyLAAChKvEipRSqmHMpDxEUQ7jHUsdio0PDRURCCgU4GCIFAYXILkDgwIAyHGBMRAzoLHQPhEQFHHBu2b/////7f//7/5//fPP/7Nv//qq3O0sx/gD2wyMHV//NExCERcxqsABAMvYDEaOLIyRpzT0HuyglBHRx1mkUUJR7CUTZw5c+5wq3ySM25GZUPxTNqlf/////////////1/6lbbylbLzdS6lRw6xVESGKVqmdjKUoCioeGiokL//NExC4RssaIAU0oAQiKgMARDAKBRXiQdZBIDGKobzYyzBRUFBkXwLICT6nqY1mM1utE5un5ktJq+3W62rQsy762SVSN1LdC6qDZxzFM0PGaCKlG58dr032eTEDUnDkH//NExDog8yq4AYdoAMHqcHYJsaD4US8QBZnAt13t/HaRxxDAGZCPHx7D1LSOUQtYnRFJU2FIkS+SZkS63T0//JpeLxQSOFxkUzc0Q540Lx0pooKMFZ5FtivUwBAUbcr4//NExAkSmfrAAc8oAOSAASDFL24xhuEkhQ7Upr//b3KqMeN9W1WmqVeXopqN+n5lcqMpNTiIwPL9FGh1tGEhZ9wBAoUeVcJng0sJFrfySpBj2rDYfpl0q5H0SgirvU7Y//NExBER0Xa0AMYElNMcnsmY7luegvPCpDD+Vr31LFf/3SUOua3nMX7HPL+hLtRujkdHZTMnv63dWHCwDSua2L0JiCp8vw2OrRqtZSxBsFZe2aoQu89kYPt4wrc9sX/K//NExBwRwJ6wAMYeTIYLY3vdogyFE+op1XNid+YLhs761AsEhUVMrOq30i6ARPElMubXikUPNBWitYy5DVQqzlBZgO2tPeFQxGFAzmVALOcqC3nyZ+henTFLLc1JYUli//NExCgScPqsAMZacPhlUeMx9NrKQNq1+r+pM2DZoCrPtLgAiGFnwoEhFu/UpC6m1uUCurlXkZKXSOWx8QxtrKn/ZQZUC13df1DmTMyCtMsRkc3dZAXJEkXEgAw0SqPt//NExDERgRqsAMZOcdtEbvzzJ40EQXOapahDb8fSuqve/YP5/BH+M1NI6BlrlLQC6FfRFqgoEcgcEU9Ckas3KlfFtr9zjPmvSLtmAKbC5jPZ9+7z4K2X5XMGHxBVQgKq//NExD4SSV6oAMZElDGfW6HBHpyZsSK1VYXOUElNYLTs0zSYu12ZfmNOVPQUWygSBqX6XfLspdkUolQAmQzmcpbl72f6vazqMDgmFwsMJhF5EkgxLRR4NV///7//T/+T//NExEcRwRKkAVgoAIq1wWY3EoQ8UNQ9KewJPWoDqV4csFFmPOtAuNWtPy4YkuVqMkUTbqlMlBljsHgYsTy+YJa+cNKi4yJk6CCb0rXjkIoVcT0LmOAFmeXX1L3/DYB2//NExFMf+wqEAZhoAA3EosvmQ5B19jI2TdzI7/X4wh4ch43RN/TLrIol1I6SqB81rR///dlMmaE4Jrdl6i8Caw9fXt475kc23Pl9RtN/iMHhb0/+KjxRpmG1nhTwblGc//NExCYaIk6oAZhAAJIFp/8IWw8ULa5kGg+LnlVEx5AsIgub38HWb/yvJwNBGHH0IAQCyCMHsHk2GEqP//8PJqXU9B6VJZHY5v+Xpd5UJIpz6d/TCcTGfoRnAANS2QmP//NExBARgZacAdk4ACoW3Jtm9nlSjy3T859jKVfO0dLtmaqaopQ9VR0OcyVVDQy44UVHf//+iF2VSLOYt20H5WWokWhNMYmakzJqgoeIq7TpUYzew4Yg7zEzDJ3PmGs5//NExB0SuR6QANPMcMv4tIcLfh6yO7nqyHlk8QVQGR/3fjejDHUaMAzz2GhwVsU7qAwiCoaMvTooJdFKxxXXNAYtwEONBYcw0vecWgiO6pKNuJKgPFmimtEs8pddnvd///NExCUSEZKQANLOlHw7bnroJyI9+g8ep5VUFYsHh4etdL2+39EsRZCqDbn4jOO61VmOWyIw49wIwDUz9zgqXOMF3LlWSIomAsoFPUVxLZ8vf+zndfvw+tVYZzUoc67///NExC8Roa6UANFGlPafXczhVwMDDjoYd1cOBuIz6wfP4Xf+pUUVquud6RhYEzbYk3IGDQgXVDTJospAYqGN///////MZdDFbzGMXqUqB4BQBDpZjGMIh0OlEg8HnQoq//NExDsSGcKIAMFKlSrfwJBRUvAV3Bvy/FJqMsBiIpMoQzgicygDGQE0s+Ws/zKVhXVoX9dXn2rPeax5rH0drtai66+279te37Eb+jfKcr6s5hRUWUKxEmmhL2psoJAa//NExEUSOaZgAVsQADrrehc14cwQc144HAyPMNK29xkFeFxj6veVe9TSmQcroX+MwRQnBc93rUvkDJ8WQLkIh07/itxBcZMUAIKDIdF7e1hKYuciYasDVgyAs3/r9nxB//NExE8gwypsAZqQAOMwDYwLsR+AcAnQcABZ///f8ZMi4aoHAaBqsZsnyAFUnxS/3////0y+RBAuGZPyIJoOdJ8N1QRKhnaOpktt0WtZvs93vr0v91aMwQ1Ibamo7Kf1//NExB8YknqwAYxoASqklHjp0z6S0f8aB7j1Lp0kBGC4J4ks20RjArxlotrNS6TSXL6aRKKNDYuny8OIySUZKMSR9kvVvmJuLXI1bw8r/ndVK2K3RqWd3T8v0l2ekLrU//NExA8RYeaEAdgoAH3HHWs8c8dZaVaKzaOVtHEgBcODXZ3RW7t9RUqu/WsunQ96tiLB1i/+0tDCyGMLCxUFmcdb/oUJQhwKGKs9Kdw9amM6pbQ5ZrwaMMSNer3WM5zB//NExBwRAMpoANPMcKb++Zq8/O3ETTApywlDRrBkJkQ6oJnf5Fjg1K/Hgs9mv28zQhD7v+o+sRHmbhE40mChDlwhafVp0gEFBSIKhxAaBU7AolseIRKDSFCUJn6LREmx//NExCsQaEJEAMvMJBexxIjSJQ0zK7Z2ld/yTna3+nI9pD9SNPQa7LeGqkhFPAiCxG3BpjWUBAwsDQNLBUNCJQCAoClTolCYa7Q4WDuIniXJbA7DXWGs9KnVcrEViYd8//NExDwRgEocAMpGJLHl/KnT3LVHviUSYT0exPzQOtQKtYcX0qBtlEQkQ0XQNsPdKN1OGskjRQEegnmzUvG5s0aUUfBp/iop/9ISFQyZBYXEZkBBIVSZCQuz///FRb+K//NExEkSiOWAAHpMcItVTEFNRTMuOTkuNVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMu//NExFEAAANIAAAAADk5LjVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMu//NExKQAAANIAAAAADk5LjVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//NExKwAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//NExKwAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV'
audio_data = 'https://ia802508.us.archive.org/5/items/testmp3testfile/mpthreetest.mp3'

str_frame = 'data:image/jpeg;base64,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'
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    global start_data
    return render_template('index.html',audio=start_data) 
    

           

@app.route('/web', methods=['GET', 'POST'])
def web():
    
    global audio_data,str_frame
    if request.method == 'POST':
        print('Incoming..')
        str_frame = request.get_json(force = True) 
        print('1')
      
        str_frame,mytext = camera.vision(str_frame)
        print('2')
        
        if mytext != '':
                print('3')
                myobj = gTTS(text=mytext, lang='en', slow=False)
                
                myobj.save("/tmp/speech.mp3")
                print('4')
                with open("/tmp/speech.mp3", "rb") as f1:
                    encoded_f1 = b64encode(f1.read())
                    audio_data = "data:audio/wav;base64,"+str(encoded_f1)[2:-1]

                str_frame = "data:image/jpeg;base64," + str(str_frame)[2:-1]
                #print(image)
                #print(audio_data)
                del encoded_f1
    return render_template('index1.html',audio=audio_data,image=str_frame)

if __name__ == '__main__':
    camera.load_saved_artifacts()
    app.run()