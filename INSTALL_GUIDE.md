# 🔧 Installation Guide - Problem Solve করার জন্য

## Problem: 'pip' is not recognized

### Solution 1: Python -m pip ব্যবহার করুন (Recommended)

Command Prompt/Terminal এ এই command run করুন:

```bash
python -m pip install -r requirements.txt
```

অথবা যদি `python` কাজ না করে:

```bash
py -m pip install -r requirements.txt
```

### Solution 2: Python Install করুন

1. https://www.python.org/downloads/ থেকে Python download করুন
2. Install করার সময় **"Add Python to PATH"** checkbox টি check করুন
3. Install complete হলে Command Prompt restart করুন
4. তারপর `pip install -r requirements.txt` run করুন

### Solution 3: Manual Installation (Step by Step)

```bash
# 1. Python check করুন
python --version
# অথবা
py --version

# 2. pip upgrade করুন
python -m pip install --upgrade pip

# 3. Dependencies install করুন
python -m pip install torch torchvision
python -m pip install streamlit
python -m pip install matplotlib numpy
python -m pip install tqdm scikit-learn reportlab
```

### Solution 4: Virtual Environment ব্যবহার করুন

```bash
# Virtual environment তৈরি করুন
python -m venv venv

# Activate করুন (Windows)
venv\Scripts\activate

# Dependencies install করুন
python -m pip install -r requirements.txt

# Run করুন
streamlit run streamlit_app.py
```

---

## ✅ Quick Fix Commands:

### Windows এ:
```bash
# Option 1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py

# Option 2
py -m pip install -r requirements.txt
py -m streamlit run streamlit_app.py
```

### যদি Python PATH এ না থাকে:
1. Python install করুন: https://www.python.org/downloads/
2. Install করার সময় "Add to PATH" check করুন
3. Command Prompt restart করুন
4. `python -m pip install -r requirements.txt` run করুন

---

## 🎯 After Installation:

Dependencies install হওয়ার পর:

```bash
streamlit run streamlit_app.py
```

Browser automatically open হবে `http://localhost:8501` এ

---

## ⚠️ Common Issues:

### Issue: "python is not recognized"
**Solution**: Python install করুন এবং PATH এ add করুন

### Issue: "pip is not recognized"  
**Solution**: `python -m pip` ব্যবহার করুন

### Issue: "Permission denied"
**Solution**: Administrator হিসেবে Command Prompt open করুন

### Issue: "Module not found" after installation
**Solution**: 
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --force-reinstall
```

---

## 📞 Need Help?

1. Python version check: `python --version`
2. pip version check: `python -m pip --version`
3. Installed packages check: `python -m pip list`

