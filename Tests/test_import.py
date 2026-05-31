import traceback
try:
    import app
    app.main()
except Exception:
    with open("crash.log", "w") as f:
        traceback.print_exc(file=f)
