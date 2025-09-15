from django.shortcuts import render, redirect,  get_object_or_404
from .models import StudentsAccount, RecommendDocument, CourseModule,RecommendCourse
from .models import ListLesson, Teacher, UserAction, ListLesson,  UserActionBook
from signup.models import Student
from .forms import DocumentUploadForm
from django.http import JsonResponse, HttpResponseBadRequest
from llama_cpp import Llama
from datetime import datetime
from django.db.models import Sum
from django.db import models
from django.utils.timezone import now
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from llama_cpp import Llama
from django.contrib import messages
from django.core.paginator import Paginator
from django.shortcuts import render, redirect
from django.core.paginator import Paginator
from django.db.models import Q
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import YouTubeSearch
from yt_dlp import YoutubeDL
from django.utils import timezone
from datetime import date
import numpy as np
import calendar
import json

'''
============================================================================
Display on the screen
============================================================================
'''
def index(request):
    email = request.session.get('user_email')
    if not email:
        return redirect('login')

    try:
        student_acc = StudentsAccount.objects.get(email=email)
    except StudentsAccount.DoesNotExist:
        return redirect('login')

    # Lấy duration theo ngày (giữ giây)
    actions = UserAction.objects.filter(user=student_acc).order_by("timestamp")
    daily_duration = {}
    for action in actions:
        date_str = action.timestamp.date().strftime("%Y-%m-%d")
        daily_duration[date_str] = daily_duration.get(date_str, 0) + action.duration

    action_dates = list(daily_duration.keys())
    action_seconds = list(daily_duration.values())  # dữ liệu giây gửi sang JS

    # Các thống kê khác
    mycourse = RecommendCourse.objects.filter(student_id=student_acc.student_id) 
    ongoing_courses = mycourse.filter(status="studying").count() 
    completed_courses = mycourse.filter(status="passed").count() 
    documents_count = RecommendDocument.objects.count() 
    total_courses = mycourse.count()
    avg_progress = round((completed_courses / total_courses) * 100, 2) if total_courses else 0
    # Lấy 5 sách gần đây sinh viên đọc
    # Lấy 5 sách gần đây sinh viên đọc
    recent_books = UserActionBook.objects.filter(
        student_id=student_acc.student_id
    ).order_by('-timestamp')[:5]

    # Lấy 5 sách khác mà sinh viên chưa đọc (ví dụ cùng source với các sách đang học)
    similar_books = UserActionBook.objects.exclude(
        student_id=student_acc.student_id
    ).order_by('-timestamp')[:5]

    return render(request, "home/home.html", {
        "user": student_acc,
        "action_dates": json.dumps(action_dates),
        "action_counts": json.dumps(action_seconds),
        "ongoing_courses": ongoing_courses,
        "completed_courses": completed_courses, 
        "documents_count": documents_count, 
        "avg_progress": avg_progress, 
        "recent_books": recent_books,
        "similar_books": similar_books,
    })


def dashboard_view(request):
    email = request.session.get('user_email')
    if not email:
        return redirect('login')

    try:
        student_acc = StudentsAccount.objects.get(email=email)
    except StudentsAccount.DoesNotExist:
        return redirect('login')

    # Lấy duration theo ngày (giữ giây)
    actions = UserAction.objects.filter(user=student_acc).order_by("timestamp")
    daily_duration = {}
    for action in actions:
        date_str = action.timestamp.date().strftime("%Y-%m-%d")
        daily_duration[date_str] = daily_duration.get(date_str, 0) + action.duration

    action_dates = list(daily_duration.keys())
    action_seconds = list(daily_duration.values())  # dữ liệu giây gửi sang JS

    # Các thống kê khác
    mycourse = RecommendCourse.objects.filter(student_id=student_acc.student_id) 
    ongoing_courses = mycourse.filter(status="studying").count() 
    completed_courses = mycourse.filter(status="passed").count() 
    documents_count = RecommendDocument.objects.count() 
    total_courses = mycourse.count()
    avg_progress = round((completed_courses / total_courses) * 100, 2) if total_courses else 0
    # Lấy 5 sách gần đây sinh viên đọc
    # Lấy 5 sách gần đây sinh viên đọc
    recent_books = UserActionBook.objects.filter(
        student_id=student_acc.student_id
    ).order_by('-timestamp')[:5]

    # Lấy 5 sách khác mà sinh viên chưa đọc (ví dụ cùng source với các sách đang học)
    similar_books = UserActionBook.objects.exclude(
        student_id=student_acc.student_id
    ).order_by('-timestamp')[:5]

    return render(request, "home/home.html", {
        "user": student_acc,
        "action_dates": json.dumps(action_dates),
        "action_counts": json.dumps(action_seconds),
        "ongoing_courses": ongoing_courses,
        "completed_courses": completed_courses, 
        "documents_count": documents_count, 
        "avg_progress": avg_progress, 
        "recent_books": recent_books,
        "similar_books": similar_books,
    })

'''
=====================================================================
My account items in sidebar
=====================================================================
'''
def account_view(request):
    email = request.session.get('user_email')

    if not email:
        return redirect('login')

    try:
        user = StudentsAccount.objects.get(email=email)
        approved = True

        if request.method == "POST":
            user.first_name = request.POST.get("first_name", user.first_name)
            user.student_id = request.POST.get("student_id", user.student_id)
            user.account_type = request.POST.get("account_type", user.account_type)
            user.phone = request.POST.get("phone", user.phone)
            dob = request.POST.get("dob")
            if dob:
                try:
                    user.birthday = datetime.strptime(dob, "%Y-%m-%d").date()
                except ValueError:
                    messages.error(request, "Ngày sinh không hợp lệ.")
            user.job = request.POST.get("career", user.job)
            user.other = request.POST.get("other", user.other)

            user.save()
            messages.success(request, "Cập nhật thông tin thành công!")

    except StudentsAccount.DoesNotExist:
        try:
            user = Student.objects.get(email=email)
            approved = False
        except Student.DoesNotExist:
            return redirect('login')

    return render(request, 'home/account.html', {
        'user': user,
        'approved': approved
    })
'''
========================================================
Calendar items on sidebar
========================================================
'''
def calendar_view(request):
    # Lấy params từ query
    year_param = request.GET.get("year")
    month_param = request.GET.get("month")

    # Nếu có thì parse int, nếu rỗng hoặc None thì dùng hôm nay
    try:
        year = int(year_param) if year_param else date.today().year
    except ValueError:
        year = date.today().year

    try:
        month = int(month_param) if month_param else date.today().month
    except ValueError:
        month = date.today().month

    # Xử lý tháng trước
    prev_month = month - 1
    prev_year = year
    if prev_month < 1:
        prev_month = 12
        prev_year -= 1

    # Xử lý tháng sau
    next_month = month + 1
    next_year = year
    if next_month > 12:
        next_month = 1
        next_year += 1

    # Lấy danh sách ngày trong tháng
    cal = calendar.Calendar(firstweekday=0)
    month_days = list(cal.itermonthdates(year, month))

    context = {
        "month": month,
        "year": year,
        "prev_month": prev_month,
        "prev_year": prev_year,
        "next_month": next_month,
        "next_year": next_year,
        "today": date.today(),
        "weekdays": ["T2", "T3", "T4", "T5", "T6", "T7", "CN"],
        "month_days": month_days,
    }
    return render(request, "home/calendar.html", context)

'''
========================================================
Courses on sidebar
========================================================
'''
def courses_view(request):
    email = request.session.get('user_email')
    if not email:
        return redirect('login')

    student = StudentsAccount.objects.get(email=email)

    # Khóa học đã đăng ký
    my_courses = RecommendCourse.objects.filter(student_id=student.student_id)

    # Nếu chưa có khóa học nào → gợi ý toàn bộ
    if not my_courses.exists():
        suggested_courses = CourseModule.objects.all()
    else:
        # Khóa học chưa đăng ký
        registered_codes = my_courses.values_list('code', flat=True)
        available_courses = CourseModule.objects.exclude(code__in=registered_codes)

        # ========================================
        # 1. Tính tổng thời lượng học trong ngày
        # ========================================
        today = now().date()
        total_today = UserAction.objects.filter(
            user=student,
            timestamp__date=today
        ).aggregate(total=Sum("duration"))["total"] or 0

        # Ngưỡng phân loại
        HIGH_ACTIVITY = 2 * 60 * 60   # > 2 giờ/ngày
        LOW_ACTIVITY = 30 * 60        # < 30 phút/ngày

        # ========================================
        # 2. Gợi ý dựa vào hành vi
        # ========================================
        if total_today >= HIGH_ACTIVITY:
            # Học chăm chỉ → gợi ý khóa học nhiều tín chỉ
            suggested_courses = available_courses.order_by("-credits")[:10]
        elif total_today <= LOW_ACTIVITY:
            # Học ít → gợi ý khóa học ngắn / ít tín chỉ
            suggested_courses = available_courses.order_by("credits")[:10]
        else:
            # Trung bình → gợi ý theo tên
            suggested_courses = available_courses.order_by("name")[:10]

    return render(request, "home/courses.html", {
        "my_courses": my_courses,
        "suggested_courses": suggested_courses
    })

# ==============================
# TF-IDF cho gợi ý sách
# ==============================
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def get_similar_books(book, all_books, top_n=5):
    """
    Return top_n similar RecommendDocument dựa trên title+author TF-IDF.
    Nếu sklearn không có, fallback bằng cách lấy sách mới nhất trừ book hiện tại.
    """
    if not book:
        return []

    all_list = list(all_books)
    if SKLEARN_AVAILABLE and len(all_list) >= 2:
        corpus = [f"{b.title} {b.author or ''}" for b in all_list]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(corpus)
        idx = all_list.index(book)
        sims = cosine_similarity(tfidf[idx], tfidf).flatten()
        indices = sims.argsort()[-(top_n + 1):][::-1]  # sort từ cao xuống
        result = []
        for i in indices:
            if i == idx:
                continue
            result.append(all_list[i])
            if len(result) >= top_n:
                break
        return result
    else:
        # fallback: newest books excluding the book itself
        return [b for b in all_list if b.id != book.id][:top_n]


# ==============================
# View chính: Documents
# ==============================
def documents_view(request):
    recommend_books = RecommendDocument.objects.all().order_by("-id")
    paginator = Paginator(recommend_books, 15)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    email = request.session.get("user_email")
    recent_books, similar_books = [], []

    if email:
        student = get_object_or_404(StudentsAccount, email=email)

        # Lấy lịch sử đọc sách của user
        recent_actions = UserActionBook.objects.filter(
            student_id=student.student_id
        ).order_by("-timestamp")[:5]
        recent_books = [a.book for a in recent_actions]

        # Nếu có sách đã đọc thì gợi ý sách tương tự cuốn mới nhất
        if recent_books:
            similar_books = get_similar_books(recent_books[0], recommend_books, top_n=5)

    if request.method == "POST":
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect("home:documents")
    else:
        form = DocumentUploadForm()

    return render(request, "home/documents.html", {
        "page_obj": page_obj,
        "form": form,
        "recent_books": recent_books,
        "similar_books": similar_books,
        "recommend_count": recommend_books.count(),
    })


# ==============================
# API: log hành vi đọc sách
# ==============================
def log_book_action(request):
    if request.method == "POST":
        email = request.session.get("user_email")
        if not email:
            return JsonResponse({"status": "fail", "message": "not logged in"}, status=400)

        student = get_object_or_404(StudentsAccount, email=email)
        student_id = student.student_id  # dùng code thay vì id

        book_id = request.POST.get("book_id")
        action = request.POST.get("action", "read")

        book = get_object_or_404(RecommendDocument, id=book_id)

        UserActionBook.objects.create(
            student_id=student_id,
            book=book,
            action=action,
            timestamp=timezone.now()
        )

        return JsonResponse({"status": "ok"})

    return JsonResponse({"status": "fail"}, status=400)


'''
=================================================================
Register button for recommend course
==================================================================
'''

def register_course(request, code):
    """API cho nút 'Đăng ký' trong template"""
    if request.method == "POST":
        email = request.session.get('user_email')
        if not email:
            return JsonResponse({"success": False, "message": "Bạn chưa đăng nhập."})

        student = StudentsAccount.objects.get(email=email)
        course = get_object_or_404(CourseModule, code=code)

        # Kiểm tra đã đăng ký chưa
        if RecommendCourse.objects.filter(student_id=student.student_id, code=course.code).exists():
            return JsonResponse({"success": False, "message": "Bạn đã đăng ký khóa học này."})

        # Tạo bản ghi mới
        RecommendCourse.objects.create(
            student_id=student.student_id,
            code=course.code,
            name=course.name,
            credits=course.credits,
            status="studying"   # hoặc "pending", tùy bạn định nghĩa
        )
        return JsonResponse({"success": True, "message": "Đăng ký thành công!"})

    return JsonResponse({"success": False, "message": "Phương thức không hợp lệ."})

'''
=================================================================
List_lesson web page
==================================================================
'''

def course_detail(request, code):
    # Lấy email từ session
    email = request.session.get('user_email')
    if not email:
        return redirect('login')

    # Lấy thông tin student
    student = get_object_or_404(StudentsAccount, email=email)

    # Lấy khóa học
    course_module = get_object_or_404(CourseModule, code=code)

    # Lấy các bài học trong khóa học, sắp xếp theo thứ tự bạn muốn
    lessons = ListLesson.objects.filter(course=course_module).order_by("id")

    # Kiểm tra xem student đã đăng ký khóa học chưa
    registered = RecommendCourse.objects.filter(
        student_id=student.student_id,
        code=course_module.code
    ).exists()

    return render(request, "home/course_detail.html", {
        "course": course_module,
        "lessons": lessons,
        "registered": registered,
        "student": student
    })

from sklearn.linear_model import LogisticRegression
import numpy as np

''' 
===============================
AI Prediction Helper for suggest what user want to watch and encoraging. 
===============================
'''
def predict_future_action(durations):
    if not durations:
        return "no_activity", 0.0

    X = np.array(durations).reshape(-1, 1)
    y = np.array([1 if d > 2*3600 else 0 for d in durations])

    # Kiểm tra có đủ 2 lớp chưa
    if len(set(y)) < 2:
        # Tự dự đoán rule-based khi dữ liệu chỉ 1 lớp
        today = durations[-1]
        prob = 1.0 if y[0] == 1 else 0.0
        return ("continue" if prob >= 0.5 else "drop", prob)

    model = LogisticRegression()
    model.fit(X, y)

    today = durations[-1]
    prob = model.predict_proba([[today]])[0][1]

    return ("continue" if prob >= 0.5 else "drop", prob)


'''
=================================================================
Notification on sidebar
==================================================================
'''
def notification_view(request):
    email = request.session.get('user_email')
    if not email:
        return redirect('login')

    student = get_object_or_404(StudentsAccount, email=email)

    # Group study time by day
    actions = UserAction.objects.filter(user=student).order_by("timestamp")
    daily_duration = {}
    for action in actions:
        date_str = action.timestamp.date().strftime("%Y-%m-%d")
        daily_duration[date_str] = daily_duration.get(date_str, 0) + action.duration

    dates = list(daily_duration.keys())
    durations = list(daily_duration.values())

    # AI prediction
    status, prob = predict_future_action(durations)

    # Message for sidebar
    if status == "continue":
        message = f"👍 Bạn có {round(prob*100, 1)}% khả năng tiếp tục học. Hãy duy trì nhé!"
    else:
        message = f"⚠️ Hôm nay bạn có nguy cơ bỏ dở ({round((1-prob)*100,1)}%). Hãy học thêm chút nữa!"

    return render(request, "home/notification.html", {
        "dates": dates,
        "durations": durations,
        "message": message,
    })

'''
=================================================================
Results on sidebar
==================================================================
'''

def results_view(request):
    email = request.session.get('user_email')
    if not email:
        return redirect('login')

    student = StudentsAccount.objects.get(email=email)
    # Lấy toàn bộ dữ liệu RecommendCourse
    my_courses = RecommendCourse.objects.filter(status="passed").count()

    # Truyền sang template
    return render(request, 'home/results.html', {'recommend_courses': my_courses})

'''
=================================================================
Teachers on topbar
==================================================================
'''

def teachers(request):
    teachers = Teacher.objects.all()
    return render(request, "home/teachers.html", {"teachers": teachers})

'''
=================================================================
Courses on topbar
==================================================================
'''

def courses_current(request):
    email = request.session.get('user_email')
    if not email:
        return redirect('login')

    student = StudentsAccount.objects.get(email=email)

    # Khóa học đã đăng ký
    my_courses = RecommendCourse.objects.filter(student_id=student)

    # Lấy danh sách mã khóa học đã đăng ký
    registered_codes = my_courses.values_list('code', flat=True)

    # Khóa học gợi ý (hiện có nhưng chưa đăng ký)
    suggested_courses = CourseModule.objects.exclude(code__in=registered_codes)

    return render(request, 'home/centers.html', {
        'my_courses': my_courses,
        'suggested_courses': suggested_courses
    })

'''
=================================================================
Helps on topbar
==================================================================
'''

def help_page(request):
    return render(request, 'home/help.html')

'''
=================================================================
Come back on login page
==================================================================
'''

def login_view(request):
    return render(request, "login/login.html")
'''
=================================================================
Chatbot on sidebar
==================================================================
'''
def chat(request):
    return render(request, "home/chat.html")
# ==============================
# Load model PhoGPT-4B-Chat GGUF khi server start
# ==============================
llm = Llama(
    model_path=r"C:\Users\dungdam\.cache\huggingface\hub\models--vinai--PhoGPT-4B-Chat-gguf\snapshots\192f8ac548e5012d28d8703111842c49fef39271\PhoGPT-4B-Chat-Q4_K_M.gguf",
    n_gpu_layers=-1,   # -1 = dùng toàn bộ GPU
    n_ctx=8192
)

# ==============================
# Hàm tìm kiếm Google offline (Selenium)
# ==============================
def search_web(query, max_results=2):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=options)

    driver.get(f"https://www.google.com/search?q={query}")

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    results = []

    for g in soup.find_all('div', class_='tF2Cxc')[:max_results]:
        title = g.find('h3').text if g.find('h3') else ''
        snippet = g.find('span', class_='aCOpRe').text if g.find('span', class_='aCOpRe') else ''
        results.append(f"{title}\n{snippet}")

    driver.quit()
    return " ".join(results)

# ==============================
# Hàm tạo reply từ model
# ==============================
def generate_reply(user_message):
    # Step 1: search web để lấy nội dung tham khảo
    web_content = search_web(user_message)

    # Step 2: feed web content vào PhoGPT-4B-Chat
    prompt = f"""
    Bạn là trợ lý AI thông minh.
    Trả lời ngắn gọn, trọng tâm, hãy trả lời chi tiết.
    Nếu câu hỏi liên quan đến nội dung bạn không biết, hoặc cần tạo hình ảnh, âm thanh, video... thì trả lời: "Tôi không thể thực hiện yêu cầu này".
    Không lặp lại câu hỏi, không hỏi lại người dùng.
    Thông tin tham khảo: {web_content}
    Câu hỏi: {user_message}
    Trả lời AI:"""

    output = llm(prompt, max_tokens=512, temperature=0.7,stop=["\n", "Người dùng:", "AI:"])
    return output['choices'][0]['text'].strip() if output.get('choices') else "[AI không trả lời được]"

# ==============================
# View API chat
# ==============================
def chat_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            message = data.get("message", "").strip()
            if not message:
                reply = "Xin vui lòng nhập tin nhắn"
            else:
                try:
                    reply = generate_reply(message)
                except Exception as e:
                    print("Lỗi generate_reply:", e)
                    reply = "[AI không trả lời được, lỗi GPU hoặc web]"
            return JsonResponse({"reply": reply})
        except Exception as e:
            print("Lỗi chat_api:", e)
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid method"}, status=400)

'''
=================================================================
Video playing pages
==================================================================
'''
def lesson_detail(request, lesson_id):
    lesson = get_object_or_404(ListLesson, id=lesson_id)
    return render(request, "home/lessons.html", {"lesson": lesson})

'''
=================================================================
Write action log  
==================================================================
'''

def log_action(request, lesson_id):
    if request.method == "POST":
        action = request.POST.get("action")
        if action in ['play', 'pause', 'back', 'done']:
            try:
                email = request.session.get('user_email')
                student_acc = StudentsAccount.objects.get(email=email)
                lesson = ListLesson.objects.get(id=lesson_id)

                # Tạo hoặc cập nhật duration trong ngày hôm nay
                today = now().date()
                
                # Lấy tất cả play chưa có duration hôm nay
                plays_today = UserAction.objects.filter(
                    user=student_acc,
                    video=lesson,
                    action="play",
                    timestamp__date=today
                ).order_by("timestamp")
                
                # Nếu action là play
                if action == "play":
                    UserAction.objects.create(
                        user=student_acc,
                        video=lesson,
                        action="play",
                        duration=0,
                        timestamp=now()
                    )
                else:
                    # Tính duration từ play gần nhất hôm nay
                    last_play = plays_today.filter(duration=0).last()
                    if last_play:
                        end_time = now()
                        duration = (end_time - last_play.timestamp).total_seconds()
                        last_play.duration = duration
                        last_play.save()
                    
                    # Tạo record cho action hiện tại
                    UserAction.objects.create(
                        user=student_acc,
                        video=lesson,
                        action=action,
                        duration=0,  # chỉ lưu duration cho play → pause/back/done sẽ update play
                        timestamp=now()
                    )

                # **Cập nhật tổng thời gian học hôm nay**
                total_duration_today = UserAction.objects.filter(
                    user=student_acc,
                    timestamp__date=today
                ).aggregate(total=models.Sum('duration'))['total'] or 0

                # Có thể lưu record tổng thời gian hôm nay vào một bảng khác nếu muốn,
                # hoặc chỉ hiển thị trực tiếp trong index.

                return JsonResponse({"status": "ok", "action": action, "today_duration": total_duration_today})

            except ListLesson.DoesNotExist:
                return JsonResponse({"status": "error", "message": "Lesson not found"}, status=404)
            except StudentsAccount.DoesNotExist:
                return JsonResponse({"status": "error", "message": "User not found"}, status=404)
        else:
            return JsonResponse({"status": "error", "message": "Invalid action"}, status=400)
    return JsonResponse({"status": "error", "message": "POST request required"}, status=400)

'''
=================================================================
Learn more customer demands thank to youtube searching actions.
==================================================================
'''

def youtube_search_view(request):
    query = request.GET.get("q")
    if not query:
        return JsonResponse({"results": []})

    # Cấu hình yt-dlp
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",  # chỉ lấy info video
    }

    results = []
    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_url = f"ytsearch10:{query}"  # lấy 10 video
            info = ydl.extract_info(search_url, download=False)
            for entry in info.get("entries", []):
                video_data = {
                    "title": entry.get("title"),
                    "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
                    "duration": entry.get("duration"),
                    "uploader": entry.get("uploader")
                }
                results.append(video_data)

                # Lưu vào cơ sở dữ liệu
                YouTubeSearch.objects.create(
                    query=query,
                    title=entry.get("title"),
                    url=f"https://www.youtube.com/watch?v={entry.get('id')}",
                    duration=entry.get("duration"),
                    uploader=entry.get("uploader"),
                    searched_at=timezone.now()
                )

        except Exception as e:
            print("YT Search error:", e)

    return JsonResponse({"results": results})





