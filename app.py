    
from src.Student_Analysis.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data=CustomData(
            
            math=float(request.form.get('math')),
            reading = float(request.form.get('reading')),
            writing= float(request.form.get('writing')),
            gender=str(request.form.get("gender")),
            race = str(request.form.get('race')),
            level_of_education = str(request.form.get('level_of_education')),
            lunch = str(request.form.get('lunch')),
            course = str(request.form.get('course')),
            
            
        )
         #this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=int(predict_pipeline.predict(final_data))
        print(final_data)
        
        
        
        
        return render_template("result.html",final_result=pred)

#execution begin
if __name__ == '__main__':
    app.run(debug=True)
