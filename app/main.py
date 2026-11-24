
import io, uuid, os
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from .analytics import profile, quality, apply_cleaning, filtered, analytics, insights, answer_question

app = FastAPI(title="InsightX", version="1.0.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
SESSIONS = {}

class CleanRequest(BaseModel):
    options: dict
class FilterRequest(BaseModel):
    filters: dict
class AskRequest(BaseModel):
    question: str

def get_session(request):
    sid = request.headers.get("x-session-id") or request.query_params.get("sid")
    if not sid or sid not in SESSIONS: raise HTTPException(400, "Dataset session not found. Upload a dataset first.")
    return sid, SESSIONS[sid]

def read_dataframe(data, filename, sheet=0):
    try:
        if filename.lower().endswith(".csv"):
            try: return pd.read_csv(io.BytesIO(data), low_memory=False)
            except UnicodeDecodeError: return pd.read_csv(io.BytesIO(data), encoding="latin-1", low_memory=False)
        if filename.lower().endswith((".xlsx",".xls")):
            return pd.read_excel(io.BytesIO(data), sheet_name=sheet)
    except Exception as e:
        raise HTTPException(400, f"Could not read dataset: {e}")
    raise HTTPException(400, "Unsupported file type. Upload CSV or Excel.")

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("app/static/index.html")

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "dataset.csv"
    if not name.lower().endswith((".csv",".xlsx",".xls")): raise HTTPException(400,"Only CSV and Excel files are supported.")
    data = await file.read()
    if len(data) > 50*1024*1024: raise HTTPException(413,"File exceeds the 50 MB limit.")
    df = read_dataframe(data,name)
    if df.empty: raise HTTPException(400,"The uploaded dataset has no rows.")
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"original":df.copy(), "current":df.copy(), "filename":name, "clean_log":[]}
    return {"sid":sid,"filename":name,"bytes":len(data),"profile":profile(df),"quality":quality(df),"preview":df.head(15).where(pd.notna(df.head(15)),None).to_dict(orient="records")}

@app.post("/api/demo")
def demo(mess=False):
    path = "sample_data/messy_sales.csv" if mess else "sample_data/sales.csv"
    df = pd.read_csv(path)
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"original":df.copy(),"current":df.copy(),"filename":os.path.basename(path),"clean_log":[]}
    return {"sid":sid,"filename":os.path.basename(path),"profile":profile(df),"quality":quality(df),"preview":df.head(15).where(pd.notna(df.head(15)),None).to_dict(orient="records")}

@app.get("/api/sheets")
async def sheets(file: UploadFile = File(...)):
    data=await file.read()
    try:
        xls=pd.ExcelFile(io.BytesIO(data))
        return {"sheets":xls.sheet_names}
    except Exception as e: raise HTTPException(400,str(e))

@app.get("/api/overview")
def overview(request: Request):
    sid,s=get_session(request); df=s["current"]
    return {"filename":s["filename"],"profile":profile(df),"quality":quality(df),"clean_log":s["clean_log"],"preview":df.head(15).where(pd.notna(df.head(15)),None).to_dict(orient="records")}

@app.post("/api/clean-preview")
def clean_preview(body: CleanRequest, request: Request):
    sid,s=get_session(request)
    new,log=apply_cleaning(s["current"],body.options)
    return {"before":profile(s["current"]),"after":profile(new),"quality_before":quality(s["current"]),"quality_after":quality(new),"log":log,"preview":new.head(10).where(pd.notna(new.head(10)),None).to_dict(orient="records")}

@app.post("/api/clean-apply")
def clean_apply(body: CleanRequest, request: Request):
    sid,s=get_session(request)
    new,log=apply_cleaning(s["current"],body.options)
    s["current"]=new
    s["clean_log"].extend(log)
    return {"profile":profile(new),"quality":quality(new),"clean_log":s["clean_log"]}

@app.post("/api/reset")
def reset(request: Request):
    sid,s=get_session(request); s["current"]=s["original"].copy(); s["clean_log"]=[]
    return {"profile":profile(s["current"]),"quality":quality(s["current"]),"message":"Restored the original uploaded dataset."}


@app.get("/api/filter-options")
def filter_options(request: Request):
    sid,s=get_session(request); df=s["current"]
    from .analytics import detect_columns
    mapping=detect_columns(df); result={}
    for role in ["category","region","product"]:
        col=mapping.get(role)
        if col:
            vals=df[col].dropna().astype(str).drop_duplicates().head(500).tolist()
            result[role]=vals
    return result

@app.post("/api/analytics")
def get_analytics(body: FilterRequest, request: Request):
    sid,s=get_session(request); df=filtered(s["current"],body.filters)
    a=analytics(df); a["insights"]=insights(df,a); a["rows_after_filter"]=len(df)
    return a

@app.post("/api/ask")
def ask(body: AskRequest, request: Request):
    sid,s=get_session(request)
    return {"answer":answer_question(s["current"],body.question)}

@app.get("/api/export/csv")
def export_csv(request: Request):
    sid,s=get_session(request); b=io.BytesIO(s["current"].to_csv(index=False).encode("utf-8-sig"))
    return StreamingResponse(b,media_type="text/csv",headers={"Content-Disposition":'attachment; filename="insightx_cleaned.csv"'})

@app.get("/api/export/xlsx")
def export_xlsx(request: Request):
    sid,s=get_session(request); b=io.BytesIO()
    with pd.ExcelWriter(b,engine="openpyxl") as w: s["current"].to_excel(w,index=False,sheet_name="Cleaned Data")
    b.seek(0)
    return StreamingResponse(b,media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",headers={"Content-Disposition":'attachment; filename="insightx_cleaned.xlsx"'})

@app.get("/api/export/report.pdf")
def export_pdf(request: Request):
    sid,s=get_session(request); df=s["current"]; a=analytics(df); q=quality(df); p=profile(df)
    b=io.BytesIO(); c=canvas.Canvas(b,pagesize=A4); width,height=A4; y=height-50
    def line(txt,size=10,bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica",size)
        c.drawString(45,y,str(txt)[:110]); y-=17
        if y<55: c.showPage(); y=height-50
    line("InsightX | Business Analytics Report",18,True); line(datetime.now().strftime("Generated %d %b %Y, %H:%M"))
    line(f"Dataset: {s['filename']}",11,True); line(f"Rows: {p['rows']:,} | Columns: {p['columns']} | Quality score: {q['score']}/100")
    line("Data quality",14,True); line(f"Issues: {q['issues']} | Missing cells: {q['missing_cells']} | Duplicate rows: {q['duplicate_rows']}")
    line("Key performance indicators",14,True)
    for k,v in a["kpis"].items(): line(f"{k.replace('_',' ').title()}: {v:,.2f}" if isinstance(v,(int,float)) else f"{k}: {v}")
    line("Business insights",14,True)
    for x in a.get("insights",[]): line(f"[{x['priority']}] {x['finding']}"); line(f"Evidence: {x['evidence']}"); line(f"Action: {x['action']}")
    line("Cleaning log",14,True)
    for x in s["clean_log"]: line(f"{x['action']}: {x['affected']} affected")
    c.save(); b.seek(0)
    return StreamingResponse(b,media_type="application/pdf",headers={"Content-Disposition":'attachment; filename="insightx_report.pdf"'})
