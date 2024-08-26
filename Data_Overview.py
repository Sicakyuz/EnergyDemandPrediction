import streamlit as st

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import plotly.express as px
from file_upload import upload_file,load_default_file


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Functions


def month_to_number(name):
    months = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12
    }
    return months.get(name)


def number_to_month(number):
    months = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct',
        11: 'Nov',
        12: 'Dec'
    }
    return months.get(number)


def create_date(month, year):
    month_number = month_to_number(month)
    date = datetime.datetime(year=int(year), month=month_number, day=1)

    return date.strftime("%Y-%m-%d")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


###########################################
def main_data():
    df = st.session_state['data']
    df["MONTH"] = df["TARIH"].dt.month
    df["TOPLAM_URETIM"] = df["LISANSSIZ_GENEL_TOPLAM"] + df["URETIM_LISANSLI"]

    # Lists

    city = df["ILLER"].unique().tolist()
    year = sorted(df["TARIH"].dt.year.unique().tolist())
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    consumption_data = ['AYDINLATMA', 'MESKEN', 'SANAYI', 'TARIMSAL_SULAMA', 'TICARETHANE', 'TUKETIM_GENEL_TOPLAM']
    production_data = ['BIYOKUTLE_LISANSSIZ', 'GUNES_LISANSSIZ', 'HIDROLIK_LISANSSIZ', 'RUZGAR_LISANSSIZ',
                       'LISANSSIZ_GENEL_TOPLAM', 'URETIM_LISANSLI']

    choice = st.sidebar.radio(
        "Analysis for:",
        ("Analysis of Energy Production Data", "Analysis of Energy Consumption Data",
         "Comparison of Energy Production Amounts and Energy Consumption Amounts")
    )
    if choice == "Analysis of Energy Production Data":
        st.title(" Analysis of Renewable Energy Resources Production ")
        p_country, p_region, p_city = st.tabs(
            ["Country-based Analysis", "Region-based Analysis", "City-based Analysis"])

        ######################Country-Based#################################
        # Year
        p_country.text("Analysis of Energy Production in Türkiye")
        col1, col2 = p_country.columns(2)
        p_co_cont = col1.container(border=True)
        p_co_cont2 = col2.container(border=True)

        bubble_year = p_co_cont.selectbox("Select Year", year, key="bubble_year")
        p_co_cont.title("#")
        p_co_cont.title("#")

        country_based_p_df = df[
            ["YEAR", 'BIYOKUTLE_LISANSSIZ', 'GUNES_LISANSSIZ', 'HIDROLIK_LISANSSIZ', 'RUZGAR_LISANSSIZ',
             'LISANSSIZ_GENEL_TOPLAM', 'URETIM_LISANSLI']]
        # country_based_p_df["MONTH"] = country_based_p_df["TARIH"].dt.month
        # country_based_p_df["MONTH"] = country_based_p_df["MONTH"].apply(number_to_month)
        country_based_p_df = country_based_p_df.groupby("YEAR").sum().reset_index()
        cbp_fig = go.Figure(data=[go.Bar(x=country_based_p_df[country_based_p_df["YEAR"] == bubble_year].columns[1:],
                                         y=country_based_p_df[country_based_p_df["YEAR"] == bubble_year].values[0][1:],
                                         marker_color='skyblue')])

        cbp_fig.update_layout(
            title=f'Energy Production in {bubble_year}',
            xaxis=dict(title='Energy Types'),
            yaxis=dict(title='Producion Data')
        )
        p_co_cont.plotly_chart(cbp_fig, use_container_width=True)

        # Year - Month

        cb2_p_df = df[
            ["TARIH", "YEAR", 'BIYOKUTLE_LISANSSIZ', 'GUNES_LISANSSIZ', 'HIDROLIK_LISANSSIZ', 'RUZGAR_LISANSSIZ',
             'LISANSSIZ_GENEL_TOPLAM', 'URETIM_LISANSLI']]
        cb2_p_df["MONTH"] = cb2_p_df["TARIH"].dt.month
        cb2_p_df["MONTH"] = cb2_p_df["MONTH"].apply(number_to_month)
        cb2_p_df.drop("TARIH", axis=1, inplace=True)
        cb2_p_df = cb2_p_df.groupby(["YEAR", "MONTH"]).sum().reset_index()
        cb2_p_df.head()
        cb2_year = p_co_cont2.selectbox("Select Year", year, key="cb2_year")
        cb2_month = p_co_cont2.selectbox("Select Month", month, key="cb2_month")

        choosen_cb2_df = cb2_p_df[(cb2_p_df["YEAR"] == cb2_year) & (cb2_p_df["MONTH"] == cb2_month)]

        cb2_fig = go.Figure(
            data=[go.Bar(x=choosen_cb2_df.columns[2:], y=choosen_cb2_df.values[0][2:], marker_color='pink')])

        cb2_fig.update_layout(
            title=f'Energy Production in {cb2_month} - {cb2_year}',
            xaxis=dict(title='Energy Types'),
            yaxis=dict(title='Producion Data'))

        p_co_cont2.plotly_chart(cb2_fig, use_container_width=True)

        ###################Region Based ###############################
        p_r_col1, p_r_col2 = p_region.columns(2)

        ## Region - Year -  Unlicensed PIE

        p_r_cont_1 = p_r_col1.container(border=True)

        pu_region_df_cols = ["BOLGE", "TARIH", "YEAR", "LISANSSIZ_GENEL_TOPLAM"]
        pu_region_df = df[pu_region_df_cols]
        pu_region_df["YEAR"] = pu_region_df["TARIH"].dt.year
        pu_region_df.drop("TARIH", axis=1, inplace=True)
        pu_region_df = pu_region_df.groupby(['BOLGE', 'YEAR']).sum().reset_index()

        pu_bar_year = p_r_cont_1.selectbox("Select Year", year, key="pu_bar_year")
        pie_u_df = pu_region_df[pu_region_df["YEAR"] == pu_bar_year]
        pie_u_fig = px.pie(pie_u_df, values="LISANSSIZ_GENEL_TOPLAM", names='BOLGE', color="BOLGE",
                           title='Rate of Unlicensed Energy Production by Regions',
                           color_discrete_map={"AKDENIZ BOLGESI": "deepskyblue",
                                               "GUNEYDOGU ANADOLU BOLGESI": "red",
                                               "EGE BOLGESI": "gold",
                                               "DOGU ANADOLU BOLGESI": "mediumorchid",
                                               "IC ANADOLU BOLGESI": "darkorange",
                                               "MARMARA BOLGESI": "mediumturquoise",
                                               "KARADENIZ BOLGESI": "forestgreen"})
        p_r_cont_1.plotly_chart(pie_u_fig, use_container_width=True)

        ## Region - Year -  Licensed PIE
        p_r_cont_2 = p_r_col2.container(border=True)

        pl_region_df_cols = ["BOLGE", "TARIH", "YEAR", "URETIM_LISANSLI"]
        pl_region_df = df[pl_region_df_cols]
        pl_region_df["YEAR"] = pl_region_df["TARIH"].dt.year
        pl_region_df.drop("TARIH", axis=1, inplace=True)
        pl_region_df = pl_region_df.groupby(['BOLGE', 'YEAR']).sum().reset_index()

        pl_bar_year = p_r_cont_2.selectbox("Select Year", year, key="pl_bar_year")
        pie_l_df = pl_region_df[pl_region_df["YEAR"] == pl_bar_year]
        pie_l_fig = px.pie(pie_l_df, values="URETIM_LISANSLI", names='BOLGE',
                           title='Rate of Licensed Energy Production by Regions', color="BOLGE",
                           color_discrete_map={"AKDENIZ BOLGESI": "deepskyblue",
                                               "GUNEYDOGU ANADOLU BOLGESI": "red",
                                               "EGE BOLGESI": "gold",
                                               "DOGU ANADOLU BOLGESI": "mediumorchid",
                                               "IC ANADOLU BOLGESI": "darkorange",
                                               "MARMARA BOLGESI": "mediumturquoise",
                                               "KARADENIZ BOLGESI": "forestgreen"})
        p_r_cont_2.plotly_chart(pie_l_fig, use_container_width=True)

        ## Region - Year -  Unlicensed
        p_r_cont_3 = p_r_col1.container(border=True)

        p_region_df_cols = ["BOLGE", "TARIH", "YEAR", 'BIYOKUTLE_LISANSSIZ', 'GUNES_LISANSSIZ', 'HIDROLIK_LISANSSIZ',
                            'RUZGAR_LISANSSIZ']
        p_region_df = df[p_region_df_cols]
        p_region_df["YEAR"] = p_region_df["TARIH"].dt.year
        p_region_df.drop("TARIH", axis=1, inplace=True)
        p_region_df = p_region_df.groupby(['BOLGE', 'YEAR']).sum().reset_index()

        p_bar_year = p_r_cont_3.selectbox("Select Year", year, key="p_bar_year")
        p_bar_region = p_r_cont_3.selectbox("Select Region", p_region_df["BOLGE"].unique(), key="p_bar_region")

        p_bar_df = p_region_df[(p_region_df["BOLGE"] == p_bar_region) & (p_region_df["YEAR"] == p_bar_year)]
        p_bar_df.drop("YEAR", axis=1, inplace=True)
        melted_df = pd.melt(p_bar_df, id_vars=["BOLGE"], var_name="ENERJI_KAYNAGI", value_name="URETIM_MIKTARI")

        fig_ = px.bar(melted_df, x="ENERJI_KAYNAGI", y="URETIM_MIKTARI", color="ENERJI_KAYNAGI",
                      title="Unlicensed Energy Production by Source",
                      labels={"URETIM_MIKTARI": "Production Quantity", "ENERJI_KAYNAGI": "Energy Source"})
        p_r_cont_3.plotly_chart(fig_, use_container_width=True)

        ### Total PIE
        p_r_cont_3 = p_r_col2.container(border=True)

        p_region_total_df_cols = ["BOLGE", "TARIH", 'LISANSSIZ_GENEL_TOPLAM', 'URETIM_LISANSLI']
        p_region_total_df = df[p_region_total_df_cols]
        p_region_total_df["YEAR"] = p_region_total_df["TARIH"].dt.year
        p_region_total_df.drop("TARIH", axis=1, inplace=True)
        p_region_total_df = p_region_total_df.groupby(['BOLGE', 'YEAR']).sum().reset_index()
        p_region_total_df["TOTAL_PRODUCTION_QUANTITY"] = p_region_total_df['LISANSSIZ_GENEL_TOPLAM'] + \
                                                         p_region_total_df[
                                                             'URETIM_LISANSLI']

        pp_bar_year = p_r_cont_3.selectbox("Select Year", year, key="pp_bar_year")
        pie_df = p_region_total_df[p_region_total_df["YEAR"] == pp_bar_year]

        pie_fig = px.pie(pie_df, values="TOTAL_PRODUCTION_QUANTITY", names='BOLGE',
                         title='Rate of Total Energy Production by Regions', color="BOLGE",
                         color_discrete_map={"AKDENIZ BOLGESI": "deepskyblue",
                                             "GUNEYDOGU ANADOLU BOLGESI": "red",
                                             "EGE BOLGESI": "gold",
                                             "DOGU ANADOLU BOLGESI": "mediumorchid",
                                             "IC ANADOLU BOLGESI": "darkorange",
                                             "MARMARA BOLGESI": "mediumturquoise",
                                             "KARADENIZ BOLGESI": "forestgreen"})
        p_r_cont_3.plotly_chart(pie_fig, use_container_width=True)

        ## Licensed - Unlicensed

        p_r_cont_4 = p_r_col2.container(border=True)
        bar_fig_u = go.Figure()

        ppl_bar_year = p_r_cont_4.selectbox("Select Year", year, key="ppl_bar_year")
        bar_df = p_region_total_df[p_region_total_df["YEAR"] == ppl_bar_year]
        bar_fig_u.add_trace(go.Bar(x=bar_df["BOLGE"], y=bar_df["LISANSSIZ_GENEL_TOPLAM"],
                                   name="Total Unlicensed Energy Production"))
        bar_fig_u.add_trace(
            go.Bar(x=bar_df["BOLGE"], y=bar_df["URETIM_LISANSLI"], name="Total Licensed Energy Production"))

        bar_fig_u.update_layout(
            title_text="Unlicensed and Licensed Production by Region",
            xaxis_title="Regions",
            yaxis_title="Production Quantity",
        )
        p_r_cont_4.plotly_chart(bar_fig_u, use_container_width=True)

        ###Multiline Total
        p_r_cont_5 = p_r_col1.container(border=True)
        p_multiline = p_r_cont_5.selectbox("Select Region", p_region_df["BOLGE"].unique(), key="p_multiline")

        p_multiline_region_df = df[df["BOLGE"] == p_multiline][
            ["TARIH", "BOLGE", 'LISANSSIZ_GENEL_TOPLAM', 'URETIM_LISANSLI']]
        p_multiline_region_df["TOTAL_PRODUCTION_QUANTITY"] = p_multiline_region_df['LISANSSIZ_GENEL_TOPLAM'] + \
                                                             p_multiline_region_df['URETIM_LISANSLI']
        p_multiline_region_df["MONTH"] = p_multiline_region_df["TARIH"].dt.month
        p_multiline_region_df["YEAR"] = p_multiline_region_df["TARIH"].dt.year
        p_multiline_region_df["MONTH"] = p_multiline_region_df["MONTH"].apply(number_to_month)
        p_multiline_region_df.drop("TARIH", axis=1, inplace=True)
        p_multiline_region_ml_df = p_multiline_region_df.groupby(['YEAR', "MONTH", "BOLGE"]).sum().reset_index()

        p_multiline_region_ml_df['MONTH'] = p_multiline_region_ml_df["MONTH"].astype(
            pd.CategoricalDtype(categories=month, ordered=True))

        c_multiline_region_ml_df = p_multiline_region_ml_df.sort_values(by=['YEAR', 'MONTH'])

        p_region_ml_fig = px.line(c_multiline_region_ml_df, x="MONTH", y="TOTAL_PRODUCTION_QUANTITY", color='YEAR',
                                  markers=True,
                                  category_orders={
                                      "MONTH": ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
                                                "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]})
        p_region_ml_fig.update_layout(
            title_text="Total Production by Month",
            xaxis_title="Month",
            yaxis_title="Total Production Quantity",
        )

        p_r_cont_5.plotly_chart(p_region_ml_fig, use_container_width=True)
        ################ City Based ###################################
        pc_col1, pc_col2 = p_city.columns(2)

        # City - Date
        pc_cont1 = pc_col1.container(border=True)

        pc_cont1.text("Analysis of Renewable Energy Resources")
        line_city = pc_cont1.selectbox("Select City", city, key="Select City")
        line_year = pc_cont1.selectbox("Select Year", year, key="Select Year")
        line_variable = pc_cont1.selectbox("Select Variable", production_data, key="line_variable")
        line_df = df[(df["ILLER"] == line_city) & (df["TARIH"].dt.year == line_year)]
        line_fig = px.line(line_df, x="TARIH", y=line_variable)
        pc_cont1.plotly_chart(line_fig, use_container_width=True)

        # Multiline
        pc_cont2 = pc_col2.container(border=True)
        pc_cont2.text("Analysis of Changes in Renewable Energy Resources")
        multiline_city = pc_cont2.selectbox("Select City", city, key="Multiline City")
        multiline_variable = pc_cont2.selectbox("Select Variable", production_data, key="multiline_variable")

        multiline_df = df[(df["ILLER"] == multiline_city)][["TARIH", multiline_variable]]
        multiline_df["YILLAR"] = multiline_df["TARIH"].dt.year
        multiline_df["AYLAR"] = multiline_df["TARIH"].dt.month

        multiline_df["AYLAR"] = multiline_df["AYLAR"].apply(number_to_month)

        multiline_fig = px.line(multiline_df, x="AYLAR", y=multiline_variable, color='YILLAR', markers=True,
                                category_orders={
                                    "AYLAR": ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
                                              "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]},
                                height=535)

        pc_cont2.plotly_chart(multiline_fig, use_container_width=True)

    elif choice == "Analysis of Energy Consumption Data":  # "Analysis of Energy Consumption Data"

        st.title(" Analysis of Energy Consumption")
        c_country, c_region, c_city = st.tabs(
            ["Country-based Analysis", "Region-based Analysis", "City-based Analysis"])

        ######################Country-Based#################################
        # Year
        c_country.text("Energy Consumption Values in Türkiye")
        cc_col1, cc_col2 = c_country.columns(2)
        c_co_cont = cc_col1.container(border=True)
        c_co_cont2 = cc_col2.container(border=True)

        cc_year = c_co_cont.selectbox("Select Year", year, key="cc_year")
        c_co_cont.title("#")
        c_co_cont.title("#")

        country_based_c_df = df[
            ["YEAR", 'AYDINLATMA', 'MESKEN', 'SANAYI', 'TARIMSAL_SULAMA', 'TICARETHANE', 'TUKETIM_GENEL_TOPLAM']]
        # country_based_p_df["MONTH"] = country_based_p_df["TARIH"].dt.month
        # country_based_p_df["MONTH"] = country_based_p_df["MONTH"].apply(number_to_month)
        country_based_c_df = country_based_c_df.groupby("YEAR").sum().reset_index()
        ccp_fig = go.Figure(data=[go.Bar(x=country_based_c_df[country_based_c_df["YEAR"] == cc_year].columns[1:],
                                         y=country_based_c_df[country_based_c_df["YEAR"] == cc_year].values[0][1:],
                                         marker_color='skyblue')])

        ccp_fig.update_layout(
            title=f'Energy Consumption',
            xaxis=dict(title='Groups Subscribers'),
            yaxis=dict(title='Consumption Amounts')
        )
        c_co_cont.plotly_chart(ccp_fig, use_container_width=True)

        # Year - Month

        cb2_c_df = df[["TARIH", "YEAR", 'AYDINLATMA', 'MESKEN', 'SANAYI', 'TARIMSAL_SULAMA', 'TICARETHANE',
                       'TUKETIM_GENEL_TOPLAM']]
        cb2_c_df["MONTH"] = cb2_c_df["TARIH"].dt.month
        cb2_c_df["MONTH"] = cb2_c_df["MONTH"].apply(number_to_month)
        cb2_c_df.drop("TARIH", axis=1, inplace=True)
        cb2_c_df = cb2_c_df.groupby(["YEAR", "MONTH"]).sum().reset_index()
        cc2_year = c_co_cont2.selectbox("Select Year", year, key="cc2_year")
        cc2_month = c_co_cont2.selectbox("Select Month", month, key="cc2_month")

        choosen_cc2_df = cb2_c_df[(cb2_c_df["YEAR"] == cc2_year) & (cb2_c_df["MONTH"] == cc2_month)]

        cc2_fig = go.Figure(
            data=[go.Bar(x=choosen_cc2_df.columns[2:], y=choosen_cc2_df.values[0][2:], marker_color='pink')])

        cc2_fig.update_layout(
            title=f'Energy Consumption by Month',
            xaxis=dict(title='Groups Subscribers'),
            yaxis=dict(title='Consumption Amounts'))

        c_co_cont2.plotly_chart(cc2_fig, use_container_width=True)

        ###################Region Based ###############################
        cr_col1, cr_col2 = c_region.columns(2)
        cr_cot1 = cr_col1.container(border=True)

        c_region_df_cols = ["TARIH", "BOLGE", 'AYDINLATMA', 'MESKEN', 'SANAYI', 'TARIMSAL_SULAMA', 'TICARETHANE']
        c_region_df = df[c_region_df_cols]
        c_region_df["YEAR"] = c_region_df["TARIH"].dt.year
        c_region_df.drop("TARIH", axis=1, inplace=True)
        c_region_df = c_region_df.groupby(['BOLGE', 'YEAR']).sum().reset_index()

        c_bar_year = cr_cot1.selectbox("Select Year", year, key="c_bar_year")
        c_bar_region = cr_cot1.selectbox("Select Region", c_region_df["BOLGE"].unique(), key="c_bar_region")

        c_bar_df = c_region_df[(c_region_df["BOLGE"] == c_bar_region) & (c_region_df["YEAR"] == c_bar_year)]
        c_melted_df = pd.melt(c_bar_df, id_vars=["BOLGE"], var_name="ENERJI_KAYNAGI", value_name="TUKETIM_MIKTARI")
        c_melted_df.drop(0, inplace=True)

        ### BAR
        fig_c = px.bar(c_melted_df, x="ENERJI_KAYNAGI", y="TUKETIM_MIKTARI", color="ENERJI_KAYNAGI",
                       title="Energy Consumption",
                       labels={"TUKETIM_MIKTARI": "Consumption Quantity", "ENERJI_KAYNAGI": "Groups Subscribers"})
        cr_cot1.plotly_chart(fig_c, use_container_width=True)

        ### PIE
        cr_cot2 = cr_col2.container(border=True)
        c_region_total_df_cols = ["BOLGE", "TARIH", 'TUKETIM_GENEL_TOPLAM']
        c_region_total_df = df[c_region_total_df_cols]
        c_region_total_df["YEAR"] = c_region_total_df["TARIH"].dt.year
        c_region_total_df.drop("TARIH", axis=1, inplace=True)
        pc_region_total_df = c_region_total_df.groupby(['BOLGE', 'YEAR']).sum().reset_index()

        c_pie_year = cr_cot2.selectbox("Select Year", year, key="c_pie_year")
        c_pie_df = c_region_total_df[c_region_total_df["YEAR"] == c_pie_year]

        c_pie_fig = px.pie(c_pie_df, values="TUKETIM_GENEL_TOPLAM", names='BOLGE',
                           title='Rate of Total Energy Consumption by Regions', color="BOLGE",
                           color_discrete_map={"AKDENIZ BOLGESI": "deepskyblue",
                                               "GUNEYDOGU ANADOLU BOLGESI": "red",
                                               "EGE BOLGESI": "gold",
                                               "DOGU ANADOLU BOLGESI": "mediumorchid",
                                               "IC ANADOLU BOLGESI": "darkorange",
                                               "MARMARA BOLGESI": "mediumturquoise",
                                               "KARADENIZ BOLGESI": "forestgreen"})
        cr_cot2.plotly_chart(c_pie_fig, use_container_width=True)

        ### MULTILINE
        cr_cot3 = cr_col1.container(border=True)

        c_ml_region = cr_cot3.selectbox("Select Region", c_region_df["BOLGE"].unique(), key="c_ml_region")

        c_multiline_region_df = df[df["BOLGE"] == c_ml_region][["TARIH", "BOLGE", "TUKETIM_GENEL_TOPLAM"]]
        c_multiline_region_df["MONTH"] = c_multiline_region_df["TARIH"].dt.month
        c_multiline_region_df["YEAR"] = c_multiline_region_df["TARIH"].dt.year
        c_multiline_region_df["MONTH"] = c_multiline_region_df["MONTH"].apply(number_to_month)
        c_multiline_region_df.drop("TARIH", axis=1, inplace=True)
        c_multiline_region_ml_df = c_multiline_region_df.groupby(['YEAR', "MONTH", "BOLGE"]).sum().reset_index()

        c_multiline_region_ml_df['MONTH'] = c_multiline_region_ml_df["MONTH"].astype(
            pd.CategoricalDtype(categories=month, ordered=True))

        c_multiline_region_ml_df = c_multiline_region_ml_df.sort_values(by=['YEAR', 'MONTH'])

        c_region_ml_fig = px.line(c_multiline_region_ml_df, x="MONTH", y="TUKETIM_GENEL_TOPLAM", color='YEAR',
                                  markers=True,
                                  category_orders={
                                      "MONTH": ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
                                                "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]})

        cr_cot3.plotly_chart(c_region_ml_fig, use_container_width=True)
        ################ City Based ###################################
        cb_col1, cb_col2 = c_city.columns(2)

        # City - Date
        cb_cot1 = cb_col1.container(border=True)

        cb_cot1.text("Analysis of Energy Consumption")
        c_line_city = cb_cot1.selectbox("Select City", city, key="Select City_c")
        c_line_year = cb_cot1.selectbox("Select Year", year, key="Select Year_c")
        c_line_variable = cb_cot1.selectbox("Select Variable", consumption_data, key="line_variable_c")
        c_line_df = df[(df["ILLER"] == c_line_city) & (df["TARIH"].dt.year == c_line_year)]
        c_line_fig = px.line(c_line_df, x="TARIH", y=c_line_variable)
        cb_cot1.plotly_chart(c_line_fig, use_container_width=True)

        # Multiline
        cb_cot2 = cb_col2.container(border=True)

        cb_cot2.text("Analysis of Changes in Energy Comsumption by Years")
        c_multiline_city = cb_cot2.selectbox("Select City", city, key="Multiline City_c")
        c_multiline_variable = cb_cot2.selectbox("Select Variable", consumption_data, key="multiline_variable_c")

        c_multiline_df = df[(df["ILLER"] == c_multiline_city)][["TARIH", c_multiline_variable]]
        c_multiline_df["YILLAR"] = c_multiline_df["TARIH"].dt.year
        c_multiline_df["AYLAR"] = c_multiline_df["TARIH"].dt.month

        c_multiline_df["AYLAR"] = c_multiline_df["AYLAR"].apply(number_to_month)

        c_multiline_fig = px.line(c_multiline_df, x="AYLAR", y=c_multiline_variable, color='YILLAR',
                                  markers=True,
                                  category_orders={
                                      "AYLAR": ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
                                                "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]})

        cb_cot2.plotly_chart(c_multiline_fig, use_container_width=True)

    else:
        st.title("Comparison of Energy Production and Energy Consumption")

        c_country, c_region, c_city = st.tabs(
            ["Country-based Analysis", "Region-based Analysis", "City-based Analysis"])

        comp_df = df[["TARIH", "YEAR", "ILLER", "BOLGE", "TOPLAM_URETIM", "TUKETIM_GENEL_TOPLAM"]]

        ###################### Country-Based #################################
        country_based_df = comp_df[["YEAR", "TOPLAM_URETIM", "TUKETIM_GENEL_TOPLAM"]].groupby(
            "YEAR").sum().reset_index()

        bar_fig_comp = go.Figure()

        bar_fig_comp.add_trace(go.Bar(x=country_based_df["YEAR"], y=country_based_df["TOPLAM_URETIM"],
                                      name="Total  Energy Production"))
        bar_fig_comp.add_trace(go.Bar(x=country_based_df["YEAR"], y=country_based_df["TUKETIM_GENEL_TOPLAM"],
                                      name="Total Energy Consumption"))

        bar_fig_comp.update_layout(
            title_text="Comparison of Energy Production and Energy Consumption in Türkiye",
            xaxis_title="Years",
            yaxis_title="Energy Quantity",
            xaxis=dict(tickmode='array', tickvals=country_based_df["YEAR"],
                       ticktext=[str(year) for year in country_based_df["YEAR"]])
        )
        c_country.plotly_chart(bar_fig_comp, use_container_width=True)

        ###################### Region-Based ##################################

        ### YEAR - REGION FOR MONTH
        comp_col1, comp_col2 = c_region.columns(2)
        comp_cot1 = comp_col1.container(border=True)

        rb_comp_df = comp_df[["BOLGE", "TARIH", "TOPLAM_URETIM", "TUKETIM_GENEL_TOPLAM"]]
        rb_comp_df["YEAR"] = rb_comp_df["TARIH"].dt.year
        rb_comp_df["MONTH"] = rb_comp_df["TARIH"].dt.month
        rb_comp_df["MONTH"] = rb_comp_df["MONTH"].apply(number_to_month)
        rb_comp_df.drop("TARIH", axis=1, inplace=True)
        rb_comp_df = rb_comp_df.groupby(["YEAR", "MONTH", "BOLGE"]).sum().reset_index()

        comp_region = comp_cot1.selectbox("Select Region", rb_comp_df["BOLGE"].unique(), key="comp_region")
        comp_year = comp_cot1.selectbox("Select YEAR", rb_comp_df["YEAR"].unique(), key="comp_year")

        comp_bar_df = rb_comp_df[(rb_comp_df["BOLGE"] == comp_region) & (rb_comp_df["YEAR"] == comp_year)]

        comp_bar_df['MONTH'] = comp_bar_df["MONTH"].astype(
            pd.CategoricalDtype(categories=month, ordered=True))

        comp_bar_df = comp_bar_df.sort_values(by=['YEAR', 'MONTH'])

        fig_bar_comp = go.Figure()

        fig_bar_comp.add_trace(go.Bar(x=comp_bar_df["MONTH"], y=comp_bar_df["TOPLAM_URETIM"],
                                      name="Total Energy Production"))
        fig_bar_comp.add_trace(go.Bar(x=comp_bar_df["MONTH"], y=comp_bar_df["TUKETIM_GENEL_TOPLAM"],
                                      name="Total Energy Consumption"))

        fig_bar_comp.update_layout(
            title_text="Comparison of Energy Production and Energy Consumption by Months",
            xaxis_title="Months",
            yaxis_title="Energy Quantity")

        comp_cot1.plotly_chart(fig_bar_comp, use_container_width=True)

        # YEAR - REGION

        comp_cot2 = comp_col2.container(border=True)

        year_region_comp_df = comp_df[["YEAR", "BOLGE", "TOPLAM_URETIM", "TUKETIM_GENEL_TOPLAM"]].groupby(
            ["YEAR", "BOLGE"]).sum().reset_index()

        comp_year_region = comp_cot2.selectbox("Select YEAR", rb_comp_df["YEAR"].unique(), key="comp_year_reg")
        comp_reg_bar_df = year_region_comp_df[year_region_comp_df["YEAR"] == comp_year_region]

        comp_year_fig = go.Figure()

        comp_year_fig.add_trace(go.Bar(x=comp_reg_bar_df["BOLGE"], y=comp_reg_bar_df["TOPLAM_URETIM"],
                                       name="Total  Energy Production"))
        comp_year_fig.add_trace(go.Bar(x=comp_reg_bar_df["BOLGE"], y=comp_reg_bar_df["TUKETIM_GENEL_TOPLAM"],
                                       name="Total Energy Consumption"))

        comp_year_fig.update_layout(
            title_text="Comparison of Energy Production and Energy Consumption by Regions",
            xaxis_title="Years",
            yaxis_title="Energy Quantity")

        comp_cot2.plotly_chart(comp_year_fig, use_container_width=True)

        ###################### City-Based ####################################

        cb_col1, cb_col2 = c_city.columns(2)

        # Year - City
        city_based_df = df[["TARIH", "ILLER", "TOPLAM_URETIM", "TUKETIM_GENEL_TOPLAM"]]
        city_based_df["YEAR"] = city_based_df["TARIH"].dt.year
        city_based_df["MONTH"] = city_based_df["TARIH"].dt.month
        city_based_df["MONTH"] = city_based_df["MONTH"].apply(number_to_month)
        city_based_df.drop("TARIH", axis=1, inplace=True)
        city_based_df_month = city_based_df
        city_based_df = city_based_df.groupby(["ILLER", "YEAR"]).sum().reset_index().drop("MONTH", axis=1)

        cb_cot1 = cb_col1.container(border=True)

        cb_yc_city = cb_cot1.selectbox("Select CITY", city_based_df["ILLER"].unique(), key="cb_yc_city")

        cb_bar_df = city_based_df[(city_based_df["ILLER"] == cb_yc_city)]

        cb_fig = go.Figure()

        cb_fig.add_trace(go.Bar(x=cb_bar_df["YEAR"], y=cb_bar_df["TOPLAM_URETIM"],
                                name="Total  Energy Production"))
        cb_fig.add_trace(go.Bar(x=cb_bar_df["YEAR"], y=cb_bar_df["TUKETIM_GENEL_TOPLAM"],
                                name="Total Energy Consumption"))

        cb_fig.update_layout(
            title_text=f"Comparison of Energy Production and Energy Consumption by Years",
            xaxis_title="Years",
            yaxis_title="Energy Quantity",
            xaxis=dict(tickmode='array', tickvals=cb_bar_df["YEAR"],
                       ticktext=[str(year) for year in cb_bar_df["YEAR"]]
                       ))

        cb_cot1.plotly_chart(cb_fig, use_container_width=True)

        # Month - City
        cb_cot2 = cb_col2.container(border=True)
        cb_yc_city_2 = cb_cot2.selectbox("Select CITY", city_based_df_month["ILLER"].unique(), key="cb_yc_city2")
        cb_yc_year = cb_cot2.selectbox("Select YEAR", city_based_df_month["YEAR"].unique(), key="cb_yc_year")

        cb_bar_df_month = city_based_df_month[
            (city_based_df_month["ILLER"] == cb_yc_city_2) & (city_based_df_month["YEAR"] == cb_yc_year)]

        cb_fig_2 = go.Figure()

        cb_fig_2.add_trace(go.Bar(x=cb_bar_df_month["MONTH"], y=cb_bar_df_month["TOPLAM_URETIM"],
                                  name="Total  Energy Production"))
        cb_fig_2.add_trace(go.Bar(x=cb_bar_df_month["MONTH"], y=cb_bar_df_month["TUKETIM_GENEL_TOPLAM"],
                                  name="Total Energy Consumption"))

        cb_fig_2.update_layout(
            title_text=f"Comparison of Energy Production and Energy Consumption by Months",
            xaxis_title="Month",
            yaxis_title="Energy Quantity")

        cb_cot2.plotly_chart(cb_fig_2, use_container_width=True)