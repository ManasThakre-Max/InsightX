
import io, re, math
from typing import Optional
import numpy as np
import pandas as pd

ALIASES = {
    "revenue": ["revenue", "sales", "amount", "total sales", "net sales", "turnover"],
    "profit": ["profit", "net profit", "gross profit", "income"],
    "cost": ["cost", "cogs", "total cost", "expense"],
    "quantity": ["quantity", "qty", "units", "units sold"],
    "order": ["order id", "order_id", "invoice", "transaction id", "transaction_id"],
    "date": ["order date", "date", "transaction date", "purchase date", "invoice date"],
    "product": ["product", "product name", "item", "sku", "product_name"],
    "category": ["category", "product category", "subcategory"],
    "region": ["region", "territory", "state", "zone"],
    "city": ["city", "town"],
    "customer": ["customer id", "customer_id", "client id", "customer"],
    "discount": ["discount", "discount %", "discount rate"],
}

def detect_columns(df):
    normalized = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    result = {}
    for role, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                result[role] = normalized[alias]
                break
        if role not in result:
            for norm, original in normalized.items():
                if any(a in norm for a in aliases if len(a) > 4):
                    result[role] = original
                    break
    return result

def safe_num(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[$,₹%]", "", regex=True), errors="coerce")

def profile(df):
    rows, cols = df.shape
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    column_info = []
    for c in df.columns:
        s = df[c]
        column_info.append({
            "name": str(c), "dtype": str(s.dtype), "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean()*100), 2),
            "unique": int(s.nunique(dropna=True)),
            "sample": [None if pd.isna(x) else str(x)[:80] for x in s.dropna().head(3).tolist()]
        })
    numeric = df.select_dtypes(include=np.number)
    stats = {}
    for c in numeric.columns:
        s = numeric[c].dropna()
        if len(s):
            stats[str(c)] = {
                "mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
                "min": round(float(s.min()), 3), "max": round(float(s.max()), 3),
                "std": round(float(s.std()), 3) if len(s)>1 else 0,
                "q1": round(float(s.quantile(.25)), 3), "q3": round(float(s.quantile(.75)), 3)
            }
    return {
        "rows": rows, "columns": cols, "column_info": column_info,
        "numeric_columns": [str(c) for c in numeric.columns],
        "categorical_columns": [str(c) for c in df.select_dtypes(include=["object","category","bool"]).columns],
        "date_candidates": [str(c) for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()],
        "duplicate_rows": int(df.duplicated().sum()), "stats": stats,
        "mapping": {k: str(v) for k,v in detect_columns(df).items()}
    }

def quality(df):
    total_cells = max(df.shape[0]*df.shape[1], 1)
    missing = int(df.isna().sum().sum())
    duplicate = int(df.duplicated().sum())
    whitespace = 0
    for c in df.select_dtypes(include=["object"]).columns:
        whitespace += int(df[c].dropna().astype(str).map(lambda x: x != x.strip()).sum())
    mapping = detect_columns(df)
    invalid_dates = 0
    if "date" in mapping:
        invalid_dates = int(pd.to_datetime(df[mapping["date"]], errors="coerce").isna().sum() - df[mapping["date"]].isna().sum())
    negative_qty = 0
    if "quantity" in mapping:
        negative_qty = int((safe_num(df[mapping["quantity"]]) < 0).sum())
    outliers = {}
    for c in df.select_dtypes(include=np.number).columns:
        s = df[c].dropna()
        if len(s) >= 4:
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3-q1
            count = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum()) if iqr else 0
            if count: outliers[str(c)] = count
    issues = missing + duplicate + whitespace + invalid_dates + negative_qty + sum(outliers.values())
    score = max(0, min(100, round(100 - (missing/total_cells*100)*.45 - (duplicate/max(len(df),1)*100)*.25 - min(whitespace/max(len(df),1)*100,100)*.1 - min((invalid_dates+negative_qty)/max(len(df),1)*100,100)*.1 - min(sum(outliers.values())/max(len(df),1)*100,100)*.1)))
    return {
        "score": score, "issues": int(issues), "missing_cells": missing,
        "missing_pct": round(missing/total_cells*100,2), "duplicate_rows": duplicate,
        "duplicate_pct": round(duplicate/max(len(df),1)*100,2), "whitespace_cells": whitespace,
        "invalid_dates": invalid_dates, "negative_quantities": negative_qty,
        "outliers": outliers
    }

def apply_cleaning(df, options):
    out = df.copy()
    log = []
    if options.get("trim_text"):
        count = 0
        for c in out.select_dtypes(include=["object"]).columns:
            before = out[c].copy()
            out[c] = out[c].map(lambda x: x.strip() if isinstance(x,str) else x)
            count += int((before.fillna("").astype(str) != out[c].fillna("").astype(str)).sum())
        log.append({"action":"Trim leading/trailing spaces","affected":count})
    if options.get("standardize_text"):
        count = 0
        for c in out.select_dtypes(include=["object"]).columns:
            if out[c].nunique(dropna=True) < 100:
                before = out[c].copy()
                out[c] = out[c].map(lambda x: re.sub(r"\s+", " ", x).title() if isinstance(x,str) else x)
                count += int((before.fillna("").astype(str) != out[c].fillna("").astype(str)).sum())
        log.append({"action":"Standardize text capitalization (categorical columns)","affected":count})
    if options.get("drop_duplicates"):
        before = len(out); out = out.drop_duplicates()
        log.append({"action":"Remove exact duplicate rows","affected":before-len(out)})
    fill = options.get("fill_missing", "none")
    if fill != "none":
        affected = 0
        for c in out.columns:
            n = int(out[c].isna().sum())
            if not n: continue
            if fill == "median" and pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].median())
            elif fill == "mode":
                modes = out[c].mode(dropna=True)
                if len(modes): out[c] = out[c].fillna(modes.iloc[0])
            elif fill == "zero" and pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(0)
            affected += n
        log.append({"action":f"Fill missing values ({fill}) where applicable","affected":affected})
    return out, log

def filtered(df, filters):
    out = df.copy()
    mapping = detect_columns(out)
    for role in ["category","region","product"]:
        col = mapping.get(role)
        selected = filters.get(role)
        if col and selected and selected != "All":
            out = out[out[col].astype(str) == selected]
    date_col = mapping.get("date")
    if date_col and (filters.get("start") or filters.get("end")):
        dates = pd.to_datetime(out[date_col], errors="coerce")
        if filters.get("start"): out = out[dates >= pd.Timestamp(filters["start"])]
        if filters.get("end"): out = out[pd.to_datetime(out[date_col], errors="coerce") <= pd.Timestamp(filters["end"])]
    return out

def analytics(df):
    m = detect_columns(df)
    result = {"mapping": {k:str(v) for k,v in m.items()}, "available": [], "kpis": {}, "charts": {}, "tables": {}}
    rev_col = m.get("revenue")
    profit_col = m.get("profit")
    cost_col = m.get("cost")
    qty_col = m.get("quantity")
    order_col = m.get("order")
    customer_col = m.get("customer")
    date_col = m.get("date")
    product_col = m.get("product")
    category_col = m.get("category")
    region_col = m.get("region")
    rev = safe_num(df[rev_col]) if rev_col else None
    profit = safe_num(df[profit_col]) if profit_col else (rev-safe_num(df[cost_col]) if rev is not None and cost_col else None)
    if rev is not None:
        result["kpis"]["revenue"] = float(rev.sum(skipna=True)); result["available"].append("revenue")
    if profit is not None:
        result["kpis"]["profit"] = float(profit.sum(skipna=True)); result["available"].append("profit")
    if order_col: result["kpis"]["orders"] = int(df[order_col].nunique(dropna=True)); result["available"].append("orders")
    else: result["kpis"]["orders"] = int(len(df))
    if qty_col:
        result["kpis"]["units"] = float(safe_num(df[qty_col]).sum(skipna=True)); result["available"].append("units")
    if customer_col:
        result["kpis"]["customers"] = int(df[customer_col].nunique(dropna=True)); result["available"].append("customers")
    if rev is not None and result["kpis"]["orders"]:
        result["kpis"]["aov"] = float(rev.sum(skipna=True)/result["kpis"]["orders"]); result["available"].append("aov")
    if rev is not None and profit is not None and rev.sum(skipna=True) != 0:
        result["kpis"]["margin"] = float(profit.sum(skipna=True)/rev.sum(skipna=True)*100); result["available"].append("margin")
    if product_col and rev is not None:
        s = pd.DataFrame({"label":df[product_col].astype(str), "value":rev}).groupby("label", dropna=False).value.sum().sort_values(ascending=False)
        result["tables"]["top_products"] = [{"name":str(k),"value":float(v)} for k,v in s.head(10).items()]
        result["tables"]["bottom_products"] = [{"name":str(k),"value":float(v)} for k,v in s.tail(10).sort_values().items()]
    if category_col and rev is not None:
        s = pd.DataFrame({"label":df[category_col].astype(str), "value":rev}).groupby("label").value.sum().sort_values(ascending=False)
        result["charts"]["category_revenue"] = [{"label":str(k),"value":float(v)} for k,v in s.items()]
    if region_col and rev is not None:
        s = pd.DataFrame({"label":df[region_col].astype(str), "value":rev}).groupby("label").value.sum().sort_values(ascending=False)
        result["charts"]["region_revenue"] = [{"label":str(k),"value":float(v)} for k,v in s.items()]
    if date_col and rev is not None:
        temp = pd.DataFrame({"date":pd.to_datetime(df[date_col],errors="coerce"),"revenue":rev})
        temp = temp.dropna().set_index("date").resample("MS").revenue.sum()
        result["charts"]["monthly_revenue"] = [{"label":d.strftime("%b %Y"),"value":float(v)} for d,v in temp.items()]
        if len(temp)>=2 and temp.iloc[-2] != 0:
            result["kpis"]["growth"] = float((temp.iloc[-1]-temp.iloc[-2])/abs(temp.iloc[-2])*100)
            result["available"].append("growth")
    if profit is not None and category_col:
        s = pd.DataFrame({"label":df[category_col].astype(str),"value":profit}).groupby("label").value.sum().sort_values(ascending=False)
        result["charts"]["category_profit"] = [{"label":str(k),"value":float(v)} for k,v in s.items()]
    return result

def insights(df, a):
    k = a["kpis"]; t = a["tables"]; c = a["charts"]; out=[]
    if "revenue" in k:
        out.append({"type":"Positive insight","priority":"Medium","finding":f"Total revenue is {k['revenue']:,.2f} across the analyzed records.","evidence":f"Revenue sum = {k['revenue']:,.2f}.","why":"This is the baseline for evaluating business performance.","action":"Compare revenue by period, category and region to identify the strongest contributors."})
    if "profit" in k and "revenue" in k:
        margin=k.get("margin",0)
        out.append({"type":"Profitability","priority":"High" if margin<10 else "Medium","finding":f"Overall profit margin is {margin:.1f}%.","evidence":f"Profit {k['profit']:,.2f} / Revenue {k['revenue']:,.2f}.","why":"Margin indicates how much revenue remains as profit under the dataset's definitions.","action":"Review low-margin products, discount levels and cost inputs before changing pricing."})
    if "growth" in k:
        g=k["growth"]
        out.append({"type":"Trend","priority":"High" if g<0 else "Low","finding":f"Latest month revenue changed by {g:.1f}% versus the previous month.","evidence":f"Computed from the latest two available monthly revenue totals.","why":"A sharp change may warrant checking seasonality, stock availability or one-off transactions.","action":"Inspect product and regional contributions for the two periods; this comparison alone does not establish cause."})
    if t.get("top_products"):
        p=t["top_products"][0]
        out.append({"type":"Product opportunity","priority":"Medium","finding":f"{p['name']} is the highest-revenue product in the current data.","evidence":f"Revenue = {p['value']:,.2f}.","why":"The product is the largest observed revenue contributor.","action":"Check its profit and inventory position before allocating more promotion."})
    if c.get("region_revenue"):
        best=c["region_revenue"][0]; worst=c["region_revenue"][-1]
        out.append({"type":"Regional opportunity","priority":"Medium","finding":f"{best['label']} leads revenue; {worst['label']} is lowest.","evidence":f"{best['label']}: {best['value']:,.2f}; {worst['label']}: {worst['value']:,.2f}.","why":"Regional variation can highlight areas for further investigation.","action":"Compare order counts, customer mix and local availability before deciding on regional investment."})
    return out

def answer_question(df, question):
    q=question.lower(); a=analytics(df); k=a["kpis"]; m=detect_columns(df)
    if any(x in q for x in ["highest profit","most profitable","top profit"]):
        p=m.get("product"); pr=m.get("profit")
        if p and pr:
            s=pd.DataFrame({"product":df[p].astype(str),"profit":safe_num(df[pr])}).groupby("product").profit.sum().sort_values(ascending=False)
            return f"{s.index[0]} has the highest total profit ({s.iloc[0]:,.2f}) in this dataset."
        return "I cannot calculate product profit: a product column and a profit column (or revenue plus cost) are required."
    if any(x in q for x in ["worst region","lowest region","weakest region"]):
        r=m.get("region"); v=m.get("revenue")
        if r and v:
            s=pd.DataFrame({"region":df[r].astype(str),"revenue":safe_num(df[v])}).groupby("region").revenue.sum().sort_values()
            return f"{s.index[0]} has the lowest revenue ({s.iloc[0]:,.2f}) among the regions."
        return "A region and revenue/sales column are needed to compare regional performance."
    if "top" in q and "product" in q:
        rows=a["tables"].get("top_products",[])
        return "Top products by revenue: " + "; ".join(f"{x['name']} ({x['value']:,.2f})" for x in rows[:5]) if rows else "I need product and revenue/sales columns to rank products."
    if "customer" in q:
        c=m.get("customer")
        return f"There are {df[c].nunique(dropna=True):,} unique customers." if c else "No customer identifier column was detected, so unique customers cannot be calculated."
    if "profit" in q and "margin" in q:
        return f"Overall profit margin is {k['margin']:.2f}%." if "margin" in k else "Profit margin needs both revenue and profit (or revenue and cost) columns."
    if "revenue" in q or "sales" in q:
        return f"Total revenue/sales is {k['revenue']:,.2f}." if "revenue" in k else "No recognized revenue/sales column was found."
    if "order" in q:
        return f"There are {k['orders']:,} orders/records based on the detected order ID." 
    return "I can answer questions about total revenue, profit margin, top products by revenue, highest-profit product, lowest-revenue region, unique customers and order counts. Ask one of those, or inspect the column mapping if a field was not detected."
