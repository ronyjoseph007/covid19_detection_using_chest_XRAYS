from django.shortcuts import render

# Create your views here.
def covipred(req):
    return render(req,'index.html')