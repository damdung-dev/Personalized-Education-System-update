from django.shortcuts import render,redirect
from django.contrib import messages
from django.http import HttpResponse
from signup.models import Student

# Create your views here.
def index(request):
    return render(request,'login/login.html')

def home_view(request):
    if request.method == 'POST':
        email = request.POST.get('id')
        password = request.POST.get('password')

        try:
            user = Student.objects.get(email=email)
        except Student.DoesNotExist:
            messages.error(request, "Email không tồn tại.")
            return render(request,'login/login.html')
        
        if user.password == password:
            request.session['user_email'] = user.email
            return redirect('dashboard/')
        else:
            messages.error(request, "Sai mật khẩu.")
            
    return render(request, 'login/login.html')

def forgotpassword_view(request):
    return render(request, 'forgotpassword/forgotpassword.html')

def signup_view(request):
    return render(request,"signup/signup.html")

