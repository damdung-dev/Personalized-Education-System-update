import random
from django.core.mail import send_mail
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from signup.models import Student
from django.contrib.auth.hashers import make_password

def index(request):
    return render(request, "forgotpassword/forgotpassword.html")

def send_test_email(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        code = random.randint(100000, 999999)

        subject = "Xác nhận quên mật khẩu qua mail"
        message = f"Chào {first_name},\n\nMã xác nhận của bạn là: {code}\n\nThân mến!"
        recipient_list = [email]

        try:
            staff = Student.objects.get(email=email, first_name=first_name)
        except Student.DoesNotExist:
            messages.error(request, "❌ Email hoặc tên không khớp.")
            return redirect('forgotpassword:forgotpassword')

        # Lưu mã và email vào session
        request.session['otp_code'] = str(code)
        request.session['email'] = email

        send_mail(subject, message, settings.EMAIL_HOST_USER, recipient_list)
        messages.success(request, "✅ Mã xác nhận đã được gửi đến email!")
        return redirect("forgotpassword:confirm_code")
    else:
        return render(request, "forgotpassword/forgotpassword.html")


def confirm_code(request):
    email = request.session.get('email')
    real_otp = request.session.get('otp_code')
    message = ''
    success = False
    allow_password_change = False

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'verify':
            entered_otp = request.POST.get('otp')
            if entered_otp == real_otp:
                message = '✅ Mã xác nhận đúng. Bạn có thể đổi mật khẩu.'
                messages.success(request,message)
                success = True
                allow_password_change = True
                request.session['verified'] = True
            else:
                message = '❌ Mã xác nhận không đúng. Vui lòng thử lại.'
                messages.error(request,message)
                request.session['verified'] = False


        elif action == 'change_password' and request.session.get('verified'):
            new_password = request.POST.get('new_password')
            retype_password = request.POST.get('retype_new_password')

            if new_password != retype_password:
                message = '❌ Mật khẩu không khớp.'
                messages.error(request,message)
            else:
                try:
                    user = Student.objects.get(email=email)
                    user.password = make_password(new_password)
                    user.save()
                    del request.session['verified']
                    message = '✅ Đổi mật khẩu thành công.'
                    messages.success(request,message)
                    success = True
                    return redirect('login')  # Điều hướng đến trang login
                except Student.DoesNotExist:
                    message = '❌ Không tìm thấy tài khoản.'
                    messages.error(request,message)
        else: 
            request.session['verified'] = False
    allow_password_change = request.session.get('verified', False)
    return render(request, 'forgotpassword/verification_code.html', {
        'message': message,
        'success': success,
        'allow_password_change': allow_password_change,
    })
