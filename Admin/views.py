from django.shortcuts import render,redirect
from Student.models import MentalStress
# Create your views here.
def login(request):
    if request.method == "POST":
        if request.method == "POST":
            usid = request.POST['username']
            pswd = request.POST['password']
            if usid == 'admin' and pswd == 'admin':
                return redirect('adminhome')

    return render(request,'adminlogin.html')

def adminhome(request):
    mental=MentalStress.objects.all()
    return render(request,"adminhome.html",{"mental":mental})