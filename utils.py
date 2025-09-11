# pip install pandas numpy
import pandas as pd
import numpy as np

def df_with_color_pixel(
    df: pd.DataFrame,
    r_col="r", g_col="g", b_col="b",
    swatch_col="Color",  # name of the swatch column
    swatch_width_px=22
) -> pd.io.formats.style.Styler:
    """
    Return a pandas Styler that shows a color 'pixel' per row using (r,g,b),
    next to all the other columns in that row.
    """

    def row_rgb_to_hex(row) -> str:
        r, g, b = row[[r_col, g_col, b_col]].astype(float)
        if np.isnan(r) or np.isnan(g) or np.isnan(b):
            return "#cccccc"  # fallback for missing RGB
        # auto-detect scale: 0–1 or 0–255
        scale = 255.0 if max(r, g, b) <= 1.0 else 1.0
        r = int(np.clip(round(r * scale), 0, 255))
        g = int(np.clip(round(g * scale), 0, 255))
        b = int(np.clip(round(b * scale), 0, 255))
        return f"#{r:02x}{g:02x}{b:02x}"

    df2 = df.copy()
    # compute hex per row
    df2["__hex__"] = df2.apply(row_rgb_to_hex, axis=1)

    # insert the (empty) swatch column at the front
    if swatch_col in df2.columns:
        df2 = df2.drop(columns=[swatch_col])
    df2.insert(0, swatch_col, "")

    # build the styled view: color the swatch cell background with the row's hex
    def style_row(row: pd.Series):
        styles = []
        for col in row.index:
            if col == swatch_col:
                styles.append(
                    f"background-color: {row['__hex__']}; "
                    f"border: 1px solid #999; "
                    f"width: {swatch_width_px}px; "
                    f"min-width: {swatch_width_px}px; "
                    f"max-width: {swatch_width_px}px;"
                )
            else:
                styles.append("")
        return styles

    styled = (
        df2.drop(columns=["__hex__"])
           .style
           .apply(style_row, axis=1)
           .set_properties(subset=[swatch_col], **{"text-align": "center"})
    )
    return styled
