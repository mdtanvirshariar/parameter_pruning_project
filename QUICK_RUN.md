# 🚀 Quick Run Guide

## ✅ Easiest Way to Run:

### Option 1: Using run_app.bat (Windows)
1. Double-click `run_app.bat`
2. Wait for browser to open automatically
3. If browser doesn't open, go to: http://localhost:8501

### Option 2: Using Python directly
```bash
python -m streamlit run streamlit_app.py
```

### Option 3: Using run_app.py
```bash
python run_app.py
```

---

## ⚠️ If App Doesn't Start:

### Check 1: Python is installed
```bash
python --version
```
Should show: `Python 3.x.x`

### Check 2: Dependencies are installed
```bash
python -c "import streamlit; print('OK')"
```

### Check 3: Port 8501 is free
If port is busy, use different port:
```bash
python -m streamlit run streamlit_app.py --server.port 8502
```

### Check 4: File exists
Make sure `streamlit_app.py` is in the current directory.

---

## 🔧 Common Issues:

### Issue: "streamlit is not recognized"
**Solution:**
```bash
python -m streamlit run streamlit_app.py
```

### Issue: "Module not found"
**Solution:**
```bash
python -m pip install streamlit torch torchvision matplotlib numpy
```

### Issue: Port already in use
**Solution:**
```bash
# Stop other streamlit instances first, or use different port
python -m streamlit run streamlit_app.py --server.port 8502
```

---

## 📝 Quick Commands:

```bash
# 1. Install dependencies (if needed)
python -m pip install -r requirements.txt

# 2. Run the app
python -m streamlit run streamlit_app.py

# 3. Open browser manually if needed
# Go to: http://localhost:8501
```

---

## ✅ Success Indicators:

When app runs successfully, you should see:
- Terminal shows: "You can now view your Streamlit app in your browser"
- Browser opens automatically (or you can open manually)
- Dashboard loads with tabs at the top

---

**Need Help?** Check the error message in terminal and share it for troubleshooting.

