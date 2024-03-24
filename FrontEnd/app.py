from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    lyrics = request.form.get('lyrics')
    #pacing = request.form.get('pacing')

    with open("../inputText.txt", 'w') as file:
        file.write(lyrics)

    # You can also pass the data to another Python function or file here

    return 'Form submitted successfully!'


if __name__ == '__main__':
    app.run(debug=True)
