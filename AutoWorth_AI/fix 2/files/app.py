# =====================================================================
# app.py -- AutoWorth AI: Streamlit front end for the Part 14 DealAdvisor
# =====================================================================
# Run from the project folder:   streamlit run app.py
#
# This file is only the user interface. Validation, feature building, prediction and the deal rating all
# live in deal_advisor.py. Nothing is trained or changed here, and no Test data is loaded.

import os

import streamlit as st

# deal_advisor.py opens its files (model, preprocessor, train_fe.csv) by relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from deal_advisor import AGE_REFERENCE_YEAR, AdvisorInputError, DealAdvisor  # noqa: E402

st.set_page_config(page_title="AutoWorth AI", page_icon="🚗")


@st.cache_resource(show_spinner="Loading the model...")
def load_advisor():
    return DealAdvisor()


try:
    adv = load_advisor()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

# Rating -> box colour (the emoji in the text carries the exact colour from the assignment)
RATING_BOX = {"great": st.success, "good": st.success, "fair": st.info,
              "slightly": st.warning, "over": st.error}


def cap(text):
    return text[:1].upper() + text[1:]


def car_summary(car):
    eng = car["engine_size"]
    eng_txt = "engine size n/a" if eng is None or eng != eng else f"{eng:g}L"
    return (f"{car['year']} {car['make']} {car['model']} | {car['transmission']} | "
            f"{car['fuel_type']} | {eng_txt} | {car['mileage']:,.0f} miles")


def show_result(res):
    st.divider()
    st.caption(car_summary(res["car"]))

    left, right = st.columns(2)
    left.metric("💰 Estimated market price", f"£{res['predicted_price']:,.0f}")
    right.metric("🏷️ Seller price", f"£{res['seller_price']:,.0f}")

    pct = res["below_market_pct"]
    detail = f" ({abs(pct):.1f}% {'below' if pct > 0 else 'above'} the estimate)" if pct else ""
    st.markdown(f"#### 💵 Difference: {res['difference_text']}{detail}")

    RATING_BOX[res["deal_key"]](f"**🚦 Deal rating: {res['deal_icon']} {res['deal_rating'].upper()}**")
    st.caption("The rating only compares the asking price with the model's estimate. "
               "It says nothing about the car's condition.")

    if res["reliability"] == "Standard":
        st.markdown(f"ℹ️ Estimate reliability: **{res['reliability']}**")
    else:
        reasons = "\n".join(f"- {cap(r)}" for r in res["reliability_reasons"])
        st.warning(f"**Estimate reliability: {res['reliability']}**\n\n{reasons}")
    if res["imputed"]:
        notes = "\n".join(f"- {cap(n)}" for n in res["imputed"])
        st.info(f"**Filled in by the advisor**\n\n{notes}")
    for w in res["warnings"]:
        st.warning(cap(w))


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.title("🚗 AutoWorth AI")
st.caption("Used car price and smart deal advisor")

makes = sorted({make for make, _ in adv.pair_counts.index})
transmissions = sorted(adv.train["transmission"].unique())
fuel_types = sorted(adv.train["fuelType"].unique())
year_min = int(adv.ranges["year"][0])
default_year = min(max(2017, year_min), AGE_REFERENCE_YEAR)

# No st.form here: the Model list has to update as soon as the Make changes.
st.subheader("Vehicle details")
col1, col2 = st.columns(2)
with col1:
    make = st.selectbox("Make", makes)
    model = st.selectbox("Model", sorted(adv.pair_counts[make].index), key=f"model_{make}")
    year = st.number_input("Year", min_value=year_min, max_value=AGE_REFERENCE_YEAR,
                           value=default_year, step=1)
    mileage = st.number_input("Mileage", min_value=0, value=30000, step=1000)
with col2:
    transmission = st.selectbox("Transmission", transmissions)
    fuel_type = st.selectbox("Fuel type", fuel_types)
    engine_size = st.number_input("Engine size (L)", min_value=0.0, value=2.0, step=0.1, format="%.1f")
    seller_price = st.number_input("Seller's asking price (£)", min_value=0, value=15000, step=250)

with st.expander("Optional details: MPG and tax"):
    st.caption("Add them for a more reliable estimate. If you leave them out, the advisor fills them in "
               "from similar cars and tells you so.")
    know_mpg = st.checkbox("I know the MPG")
    mpg = st.number_input("MPG", min_value=1.0, value=45.0, step=0.5, disabled=not know_mpg)
    know_tax = st.checkbox("I know the annual tax")
    tax = st.number_input("Tax (£ per year)", min_value=0, value=145, step=5, disabled=not know_tax)

if st.button("Estimate deal", type="primary"):
    car = {"make": make, "model": model, "year": year, "mileage": mileage,
           "transmission": transmission, "fuel_type": fuel_type, "engine_size": engine_size,
           "mpg": mpg if know_mpg else None, "tax": tax if know_tax else None}
    result = None
    try:
        result = adv.advise(car, seller_price)
    except AdvisorInputError as err:          # must come first: it is a subclass of ValueError
        for message in err.errors:
            st.error(cap(message))
    except ValueError as err:                 # e.g. an invalid asking price
        st.error(cap(str(err)))
    if result is not None:
        show_result(result)

with st.expander("About this estimate"):
    st.markdown(
        "- The price is a model estimate based on historical used-car listings.\n"
        "- The data does not include condition, location, service history, trim or optional extras.\n"
        f"- The data ends at model year {AGE_REFERENCE_YEAR}, so newer cars cannot be valued. "
        "Only make and model combinations seen in the training data are accepted.\n"
        "- Errors are larger for cheap cars and for expensive or rare cars. The ±5% and ±10% rating bands "
        "are about as wide as the model's typical error, so treat the rating as a screening signal, "
        "not a guarantee.\n"
        "- If you leave out MPG or tax, they are replaced by the median of similar training cars."
    )
