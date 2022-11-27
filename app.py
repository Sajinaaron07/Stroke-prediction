from util import *
from werkzeug.datastructures import MultiDict
####################### Flask settings #######################
app = Flask(__name__)

####################### Home Page ############################
@app.route("/", methods=['GET'])
@app.route("/home", methods=['GET'])
def home():
   return render_template("index.html")

########################## Results Page #######################
@app.route("/result", methods = ["GET","POST"])
def predict():
	if request.method == 'POST':


		#bro ithu tha na potta code

		mutableData = MultiDict(request.form) #intha request.form immutableDict bro ithu namma explicit ah MultiDict ah matharom mela import pannirupom paru
		h = mutableData.pop("Height") #height oda values eduakarom bro 
		w = mutableData.pop("weight") # inga weight oda values eduakarom bro and also inth weight and height anth dict la irunthu remove aurum pop panra nala
		BMI = round(int(w[0]) / int(h[0]) / int(h[0]) * 10000) #inga tha bmi ya calculate panrom
		mutableData.add('bmi',str(2)) #antha bmi value va mutableData ngra Dict la insert panrom bro avalothaeee
		feature_dict = mutableData

		#ithu varaikum bruh....W


		X = []
		for col in features:
			if col in ['hypertension', 'heart_disease']:
				X.append(int(feature_dict[col]))
			else:
				X.append(feature_dict[col])
		proba = round(predict_class([X]),2)
		return render_template("index.html", proba=proba)
####################################################

@app.route("/about")
def about():
	return render_template("about.html")

if __name__ == '__main__':
	app.run(debug=True)