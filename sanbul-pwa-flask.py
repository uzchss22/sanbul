import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
import os

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
cat_attribs = ['month', 'day']


class LabForm(FlaskForm):
    longitude      = StringField('longitude(1-7)',                    validators=[DataRequired()])
    latitude       = StringField('latitude(1-7)',                     validators=[DataRequired()])
    month          = StringField('month(01-Jan ~ 12-Dec)',            validators=[DataRequired()])
    day            = StringField('day(00-sun ~ 06-sat, 07-hol)',      validators=[DataRequired()])
    avg_temp       = StringField('avg_temp',                          validators=[DataRequired()])
    max_temp       = StringField('max_temp',                          validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed',                    validators=[DataRequired()])
    avg_wind       = StringField('avg_wind',                          validators=[DataRequired()])
    submit         = SubmitField('Submit')


# ── pipeline 로드 (pkl 있으면 로드, 없으면 데이터에서 재구성) ──────────────────
def load_pipeline():
    if os.path.exists('full_pipeline.pkl'):
        return joblib.load('full_pipeline.pkl')
    # pkl 없을 경우 학습 데이터로 재구성
    fires = pd.read_csv('./sanbul2district-divby100.csv', sep=',')
    fires['burned_area'] = np.log(fires['burned_area'] + 1)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, _ in split.split(fires, fires['month']):
        strat_train = fires.loc[train_idx]
    train_features = strat_train.drop(['burned_area'], axis=1)
    num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs),
    ])
    pipeline.fit(train_features)
    return pipeline


full_pipeline = load_pipeline()
model = tf.keras.models.load_model('fires_model.keras')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        try:
            data = pd.DataFrame([[
                float(form.longitude.data),
                float(form.latitude.data),
                form.month.data,
                form.day.data,
                float(form.avg_temp.data),
                float(form.max_temp.data),
                float(form.max_wind_speed.data),
                float(form.avg_wind.data),
            ]], columns=['longitude', 'latitude', 'month', 'day',
                         'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind'])

            prepared   = full_pipeline.transform(data)
            pred_log   = float(model.predict(prepared)[0][0])
            pred_area  = round(float(np.exp(pred_log) - 1), 2)

            return render_template('result.html', prediction=pred_area)
        except Exception as e:
            return render_template('prediction.html', form=form, error=str(e))

    return render_template('prediction.html', form=form)


if __name__ == '__main__':
    app.run(port=5001, debug=True)
