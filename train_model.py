"""Train the model on highfy_sales.csv and export its predictions to data.js (used by index.html)."""
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("highfy_sales.csv", parse_dates=["date"])
df["ym"] = df.date.dt.to_period("M").astype(str)
g = df.groupby(["area", "product", "ym"]).quantity.sum().unstack(fill_value=0).stack().reset_index(name="units")
g["month"] = g.ym.str[5:].astype(int)
enc = lambda d: pd.get_dummies(d[["area", "product"]], dtype=int).assign(month=d.month)
X = enc(g)
Xtr, Xte, ytr, yte = train_test_split(X, g.units, test_size=0.2, random_state=42)
m = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
r2 = r2_score(yte, m.predict(Xte))

grid = pd.MultiIndex.from_product([g.area.unique(), g["product"].unique(), range(1, 13)],
                                  names=["area", "product", "month"]).to_frame(index=False)
grid["units"] = m.predict(enc(grid).reindex(columns=X.columns, fill_value=0))
s = grid.groupby(["area", "product"]).units.mean().round(1)

areas = df.groupby("area")[["latitude", "longitude"]].mean().round(5)
data = {
    "r2": round(r2, 2), "orders": len(df),
    "products": sorted(df["product"].unique().tolist()),
    "areas": {a: {"lat": r.latitude, "lng": r.longitude} for a, r in areas.iterrows()},
    "scores": {a: s[a].to_dict() for a in areas.index},
}
open("data.js", "w", encoding="utf-8").write("const DATA = " + json.dumps(data) + ";")
print("data.js written | R2:", data["r2"])
