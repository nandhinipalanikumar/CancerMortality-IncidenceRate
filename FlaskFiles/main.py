import pickle
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = pickle.load(open('randomforestmodel.pkl', 'rb'))
label_encoder = LabelEncoder()

df_NV = pd.read_csv("C:/Users/Jayalakshmi/Documents/CSV files/NMinputs.csv")

label_encoder.fit(df_NV['Province1'])

def encode_input_data(data):
    data['Province'] = label_encoder.transform(data['Province'])
    return data

@app.route('/')
def welcome():
    return render_template('index-1.html')

@app.route('/predict', methods=['POST'])
def predict():
    index_1 = float(request.form["index"])
    FIPS_2 = float(request.form["FIPS"])
    abc = float(request.form["Age-Adjusted Incidence Rate(Ê) - cases per 100,000"])
    lowerabc = float(request.form["Lower 95% Confidence Interval"])
    upperabc = float(request.form["Upper 95% Confidence Interval"])
    average = float(request.form["Average Annual Count"])
    rates = float(request.form["Recent 5-Year Trend (ˆ) in Incidence Rates"])
    lowerconf = float(request.form["Lower 95% Confidence Interval.1"])
    upperconf = float(request.form["Upper 95% Confidence Interval.1"])
    state = request.form["State"]
    province = request.form["Province"]

    state_encoded = pd.get_dummies([state], prefix='State').astype(bool)
    encode = encode_input_data(pd.DataFrame({'Province': [province]}))

    features = pd.DataFrame({
        'index': [index_1],
        'FIPS': [FIPS_2],
        'Age-Adjusted Incidence Rate(Ê) - cases per 100,000': [abc],
        'Lower 95% Confidence Interval': [lowerabc],
        'Upper 95% Confidence Interval': [upperabc],
        'Average Annual Count': [average],
        'Recent 5-Year Trend (ˆ) in Incidence Rates': [rates],
        'Lower 95% Confidence Interval.1': [lowerconf],
        'Upper 95% Confidence Interval.1': [upperconf],
        'State': [state]
    })

    total = pd.concat([features, state_encoded, encode], axis=1)

    missing_columns = set(df_NV.columns) - set(total.columns)
    for column in missing_columns:
        total[column] = False

    total = total[df_NV.columns]

    prediction = model.predict(total)
    prediction = int(prediction[0])

    if prediction == 0:
        return render_template('index-1.html', predict="Stable")
    elif prediction == 1:
        return render_template('index-1.html', predict="Falling")
    elif prediction == 2:
        return render_template('index-1.html', predict="*")
    elif prediction == 3:
        return render_template('index-1.html', predict="Rising")
    elif prediction == 4:
        return render_template('index-1.html', predict="_")
    elif prediction == 5:
        return render_template('index-1.html', predict="__")
    else:
        return render_template('index-1.html', predict="Non")

if __name__ == "__main__":
    app.run(debug=False)
