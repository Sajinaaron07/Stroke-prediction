from util import *
from werkzeug.datastructures import MultiDict
import db


####################### Flask settings #######################
app = Flask(__name__)



####################### Home Page ############################

@app.route("/test")
def test():
    db.db.app1.insert_one({"name": "John"})
    return "Connected to the data base!"



@app.route("/", methods=['GET'])
@app.route("/home", methods=['GET'])
def home():
   return render_template("index.html")


########################## Results Page #######################
@app.route("/result", methods = ["GET","POST"])
def predict():
	if request.method == 'POST':

		Gender  = request.args.get('gender')
		Age  = request.args.get('age')
		Hypertension  = request.args.get('hypertension')
		HeartDie  = request.args.get('heart_disease')
		Mart  = request.args.get('ever_married')
		Work  = request.args.get('work_type')
		resd  = request.args.get('Residence_type')
		gulc  = request.args.get('avg_glucose_level')
		ht  = request.args.get('Height')
		wt  = request.args.get('weight')
		smk  = request.args.get('smoking_status')

		



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

		myQuery = {
			"gender":Gender,
			"age":Age,
			"Hypertension":Hypertension,
			"HeartDie" : HeartDie,
			"Mart" : Mart,
			"Work" : Work,
			"resd" : resd,
			"gulc" : gulc,
			"heart" : ht,
			"wt" : wt,
			"smk" : smk,
			"proba":proba
		}
		
		db.db.test.insert_one(myQuery)
		return render_template("index.html", proba=proba)
####################################################

@app.route("/about")
def about():
	return render_template("about.html")

if __name__ == '__main__':
	app.run(debug=True)