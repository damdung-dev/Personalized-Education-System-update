from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from signup.models import Student  # Thay vì Student
from django.db import IntegrityError, DatabaseError
import datetime

def index(request):
    return render(request, 'signup/signup.html')

def home_view(request):
    if request.method == "POST":
        if request.POST.get('register') == "back":
            return render(request, 'login/login.html')

        if request.POST.get('register') == "create":
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            gender = request.POST.get('gender')
            email = request.POST.get('email')
            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm-password')

            if password != confirm_password:
                messages.error(request, "Mật khẩu không trùng khớp")
                return render(request, 'signup/signup.html')

            if gender not in ["Male", "Female"]:
                messages.error(request, "Vui lòng chọn giới tính")
                return render(request, 'signup/signup.html')

            gender_value = 1 if gender == "Male" else 0

            try:
                Student.objects.create(
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    password=password,
                    gender=gender_value,
                    created_date=datetime.datetime.now()
                )

                send_mail(
                    'Đăng ký thành công',
                    'Cảm ơn bạn đã đăng ký. Hệ thống sẽ duyệt và phản hồi trong vài ngày tới.',
                    settings.DEFAULT_FROM_EMAIL,
                    [email],
                    fail_silently=False,
                )

                request.session['user_email'] = email
                return redirect('account')  # tên URL patterns bạn khai báo
            except IntegrityError:
                messages.error(request, "Email đã được sử dụng")
            except DatabaseError as e:
                messages.error(request, f"Lỗi cơ sở dữ liệu: {str(e)}")
            except Exception as e:
                messages.error(request, f"Lỗi không xác định: {str(e)}")

    return render(request, 'signup/signup.html')
