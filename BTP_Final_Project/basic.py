from flask import Flask ,render_template,request,Response
from new import myinput_network
from os.path import join, dirname, realpath
import matplotlib.pyplot as plt
import pandas as pd
app = Flask(__name__)
r=[] 
xs=[]
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/signup')
def signup():
    return render_template('signup.html')
@app.route('/signup', methods=['POST'])
@app.route('/thankyou', methods=['POST'])
def thankyou():
    text = request.form['text']
    result,x=myinput_network(text)
    result=[r*100 for r in result]
    r.extend(result)
    xs.extend(x)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.set_ylim(0,100)
    x=['obscenity','insulting','spitefulness','vilification','antagonistic','threatning']
    ax.bar(x, height= result)
    path= join(dirname(realpath(__file__)), 'static/images/')
    fig.savefig(path+"myimg.png",bbox_inches = 'tight')
    img='../static/images/myimg.png'
    return render_template("thankyou.html",result=result,img=img)

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html")
@app.errorhandler(405)
def page_not_found(e):
    return render_template("404.html")
if __name__ == '__main__':
    app.run(debug=True)
    