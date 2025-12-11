# 🚀 راهنمای اجرای ربات

## روش 1: اجرا با PowerShell (توصیه می‌شود)

1. PowerShell را باز کنید
2. به پوشه پروژه بروید:
   ```powershell
   cd C:\Users\nft_filipping\Desktop\pumpdum
   ```

3. ربات را اجرا کنید:
   ```powershell
   .\run_bot.ps1
   ```

یا:
```powershell
powershell -ExecutionPolicy Bypass -File run_bot.ps1
```

## روش 2: اجرا مستقیم با Python

```bash
python main.py
```

## روش 3: اجرا با فایل batch

```bash
start.bat
```

---

## 📊 خروجی که می‌بینید:

```
🔄 Monitoring Cycle #1 - 14:30:15
   Checking 38 coins...
   📊 Batch 1/4: Checking 10 coins...
      ✓ Checked 10/38 coins...
   📊 Batch 2/4: Checking 10 coins...
      ✓ Checked 20/38 coins...
   ✅ Cycle complete in 15.2s - Checked 38 coins
   ℹ️  No alerts in this cycle
   ⏳ Waiting 10s before next cycle...
```

## 🔍 اگر ربات گیر کرد:

1. **Timeout**: هر کوین 30 ثانیه timeout دارد
2. **Batch Processing**: کوین‌ها 10 تا 10 تا پردازش می‌شوند
3. **Logging**: پیشرفت در هر لحظه نمایش داده می‌شود

## ⚠️ مشکلات رایج:

### اگر PowerShell خطا داد:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### اگر Python پیدا نشد:
- مطمئن شوید Python نصب است
- PATH را بررسی کنید

### اگر ربات خیلی کند است:
- در `config.py` تعداد کوین‌ها را کاهش دهید:
  ```python
  MAX_COINS_TO_MONITOR = 20  # کمتر = سریع‌تر
  ```

---

**موفق باشید! 🚀**

