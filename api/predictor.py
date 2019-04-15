from nltk import ngrams
from sklearn.externals import joblib
from collections import OrderedDict
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_feature_list():
    response = S3.get_object(Bucket=BUCKET_NAME, Key=FEATURE_LIST_FILE)
    json_str = response['Body'].read()
    return json.loads(json_str)

def load_model(key):
    # Load model from S3 bucket
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    model_str = response['Body'].read()
    model = pickle.loads(model_str)
    return model

S3 = boto3.client('s3', region_name='eu-central-1')
BUCKET_NAME = 'predictorr'
MODEL_FILE_NAME = 'decision_tree.sav'
FEATURE_LIST_FILE = 'feature_list.json'
MODEL_ = load_model(MODEL_FILE_NAME)
FEATURES_ = load_feature_list()

def prepare_input_data(surname, conservative, opposition):
    grams = ngrams(surname, 2)
    features = FEATURES_
    inp = np.zeros((1, len(features)))
    for gram in grams:
        bigram = str(gram[0]) + str(gram[1])
        try:
            index = features.index(bigram)
            inp[0][index] = 1
        except:
            pass
    conv_index = features.index("conservative")
    ops_index = features.index("opposition")
    inp[0][conv_index] = conservative
    inp[0][ops_index] = (1 - opposition)
    return inp


@app.route('/', methods=['POST', 'GET', 'OPTIONS'])
def index():
    # Parse request body for model input
    body_dict = request.get_json(silent=True)
    try:
        surname = body_dict['surname']
        conserv = body_dict['conservative']
        opposition = body_dict['opposition']
        inp = prepare_input_data(surname, int(conserv), int(opposition))
    # Load model
        model = MODEL_
        prediction = model.predict_proba(inp)
        result = {}
        for i in range(0, len(model.classes_)):
            result[model.classes_[i]] = prediction[0][i]
        result = {'prediction': result}
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except:
        return  jsonify({'result': True})
if __name__ == '__main__':
    # listen on all IPs
    app.run(host='0.0.0.0')
