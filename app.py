import streamlit as st
import pandas as pd
import joblib
import numpy as np


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IPL Match Winner Prediction",
    page_icon="🏏",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(
        135deg,
        #e8f5e9,
        #f1f8e9
    );
    border: 1px solid #c8e6c9;
    text-align: center;
    margin-top: 25px;
}

.result-label {
    font-size: 18px;
    color: #555;
    margin-bottom: 8px;
}

.winner {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 10px;
}

.probability {
    font-size: 21px;
    font-weight: 600;
    margin-top: 10px;
}

.details-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7fa;
    border: 1px solid #e0e0e0;
    margin-top: 20px;
}

.detail-title {
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

.detail-item {
    font-size: 17px;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🏏 IPL Match Winner Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the winning team using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "ipl_winner_model.pkl"
    )


@st.cache_data
def load_data():

    return pd.read_csv(
        "matches_engineered.csv"
    )


@st.cache_resource
def load_feature_columns():

    return joblib.load(
        "feature_columns.pkl"
    )


@st.cache_resource
def load_winner_encoder():

    return joblib.load(
        "label_encoder.pkl"
    )


model = load_model()

matches = load_data()

feature_columns = load_feature_columns()

winner_encoder = load_winner_encoder()


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

missing_columns = [
    column
    for column in feature_columns
    if column not in matches.columns
]

if missing_columns:

    st.error(
        f"Missing model features: {missing_columns}"
    )

    st.stop()


# =========================================================
# TEAM LIST
# =========================================================

teams = sorted(
    set(matches["team1"].dropna())
    |
    set(matches["team2"].dropna())
)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🏏 Enter Match Details'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    team1 = st.selectbox(
        "Team 1",
        teams
    )


with col2:

    team2 = st.selectbox(
        "Team 2",
        teams
    )


# =========================================================
# VENUE
# =========================================================

venue_list = sorted(
    matches["venue"]
    .dropna()
    .unique()
)

venue = st.selectbox(
    "📍 Venue",
    venue_list
)


# =========================================================
# TOSS WINNER
# =========================================================

toss_winner = st.selectbox(
    "🪙 Toss Winner",
    [team1, team2]
)


# =========================================================
# TOSS DECISION
# =========================================================

toss_decision = st.selectbox(
    "🎯 Toss Decision",
    ["bat", "field"]
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔮 Predict Winner",
    use_container_width=True
)


if predict_button:

    # =====================================================
    # VALIDATION
    # =====================================================

    if team1 == team2:

        st.error(
            "Please select two different teams."
        )

        st.stop()


    # =====================================================
    # TEAM HISTORY
    # =====================================================

    team1_history = matches[
        (matches["team1"] == team1)
        |
        (matches["team2"] == team1)
    ].copy()


    team2_history = matches[
        (matches["team1"] == team2)
        |
        (matches["team2"] == team2)
    ].copy()


    if team1_history.empty or team2_history.empty:

        st.error(
            "Historical data is not available for "
            "the selected team."
        )

        st.stop()


    # =====================================================
    # SORT BY DATE
    # =====================================================

    if "date" in matches.columns:

        team1_history["date"] = pd.to_datetime(
            team1_history["date"],
            errors="coerce"
        )

        team2_history["date"] = pd.to_datetime(
            team2_history["date"],
            errors="coerce"
        )

        team1_history = team1_history.sort_values(
            "date"
        )

        team2_history = team2_history.sort_values(
            "date"
        )


    latest_team1 = team1_history.iloc[-1]

    latest_team2 = team2_history.iloc[-1]


    # =====================================================
    # WIN PERCENTAGE
    # =====================================================

    if latest_team1["team1"] == team1:

        team1_win_pct = latest_team1[
            "team1_win_pct"
        ]

    else:

        team1_win_pct = latest_team1[
            "team2_win_pct"
        ]


    if latest_team2["team1"] == team2:

        team2_win_pct = latest_team2[
            "team1_win_pct"
        ]

    else:

        team2_win_pct = latest_team2[
            "team2_win_pct"
        ]


    # =====================================================
    # LAST 5 WINS
    # =====================================================

    if latest_team1["team1"] == team1:

        team1_last5_wins = latest_team1[
            "team1_last5_wins"
        ]

    else:

        team1_last5_wins = latest_team1[
            "team2_last5_wins"
        ]


    if latest_team2["team1"] == team2:

        team2_last5_wins = latest_team2[
            "team1_last5_wins"
        ]

    else:

        team2_last5_wins = latest_team2[
            "team2_last5_wins"
        ]


    # =====================================================
    # HEAD TO HEAD
    # =====================================================

    h2h = matches[
        (
            (matches["team1"] == team1)
            &
            (matches["team2"] == team2)
        )
        |
        (
            (matches["team1"] == team2)
            &
            (matches["team2"] == team1)
        )
    ]


    team1_h2h = (
        h2h["winner"] == team1
    ).sum()


    team2_h2h = (
        h2h["winner"] == team2
    ).sum()


    # =====================================================
    # TOSS WIN PERCENTAGE
    # =====================================================

    team1_toss_wins = (
        team1_history["toss_winner"] == team1
    ).sum()


    team2_toss_wins = (
        team2_history["toss_winner"] == team2
    ).sum()


    team1_toss_win_pct = (
        team1_toss_wins /
        len(team1_history)
    ) * 100


    team2_toss_win_pct = (
        team2_toss_wins /
        len(team2_history)
    ) * 100


    # =====================================================
    # VENUE WIN PERCENTAGE
    # =====================================================

    venue_matches = matches[
        matches["venue"] == venue
    ]


    team1_venue_matches = venue_matches[
        (venue_matches["team1"] == team1)
        |
        (venue_matches["team2"] == team1)
    ]


    team2_venue_matches = venue_matches[
        (venue_matches["team1"] == team2)
        |
        (venue_matches["team2"] == team2)
    ]


    if len(team1_venue_matches) > 0:

        team1_venue_win_pct = (
            (
                team1_venue_matches["winner"]
                == team1
            ).sum()
            /
            len(team1_venue_matches)
        ) * 100

    else:

        team1_venue_win_pct = 0.0


    if len(team2_venue_matches) > 0:

        team2_venue_win_pct = (
            (
                team2_venue_matches["winner"]
                == team2
            ).sum()
            /
            len(team2_venue_matches)
        ) * 100

    else:

        team2_venue_win_pct = 0.0


    # =====================================================
    # TOSS DECISION SUCCESS RATE
    # =====================================================

    decision_matches = matches[
        matches["toss_decision"] == toss_decision
    ]


    if len(decision_matches) > 0:

        decision_success = (
            decision_matches["toss_winner"]
            ==
            decision_matches["winner"]
        ).sum()


        toss_decision_success_rate = (
            decision_success /
            len(decision_matches)
        ) * 100

    else:

        toss_decision_success_rate = 50.0


    # =====================================================
    # TEAM FEATURE FUNCTION
    # =====================================================

    def get_team_feature(
        team,
        team1_column,
        team2_column
    ):

        history = matches[
            (matches["team1"] == team)
            |
            (matches["team2"] == team)
        ].copy()


        if history.empty:

            return 0.0


        if "date" in history.columns:

            history["date"] = pd.to_datetime(
                history["date"],
                errors="coerce"
            )

            history = history.sort_values(
                "date"
            )


        latest = history.iloc[-1]


        if latest["team1"] == team:

            return latest[team1_column]

        else:

            return latest[team2_column]


    # =====================================================
    # TEAM STATISTICS
    # =====================================================

    team1_batting_avg = get_team_feature(
        team1,
        "team1_batting_avg",
        "team2_batting_avg"
    )


    team2_batting_avg = get_team_feature(
        team2,
        "team1_batting_avg",
        "team2_batting_avg"
    )


    team1_bowling_economy = get_team_feature(
        team1,
        "team1_bowling_economy",
        "team2_bowling_economy"
    )


    team2_bowling_economy = get_team_feature(
        team2,
        "team1_bowling_economy",
        "team2_bowling_economy"
    )


    team1_powerplay_runs = get_team_feature(
        team1,
        "team1_powerplay_runs",
        "team2_powerplay_runs"
    )


    team2_powerplay_runs = get_team_feature(
        team2,
        "team1_powerplay_runs",
        "team2_powerplay_runs"
    )


    team1_death_runs = get_team_feature(
        team1,
        "team1_death_runs",
        "team2_death_runs"
    )


    team2_death_runs = get_team_feature(
        team2,
        "team1_death_runs",
        "team2_death_runs"
    )


    team1_boundary_pct = get_team_feature(
        team1,
        "team1_boundary_pct",
        "team2_boundary_pct"
    )


    team2_boundary_pct = get_team_feature(
        team2,
        "team1_boundary_pct",
        "team2_boundary_pct"
    )


    team1_run_rate = get_team_feature(
        team1,
        "team1_run_rate",
        "team2_run_rate"
    )


    team2_run_rate = get_team_feature(
        team2,
        "team1_run_rate",
        "team2_run_rate"
    )


    team1_wicket_ability = get_team_feature(
        team1,
        "team1_wicket_ability",
        "team2_wicket_ability"
    )


    team2_wicket_ability = get_team_feature(
        team2,
        "team1_wicket_ability",
        "team2_wicket_ability"
    )


    # =====================================================
    # CREATE INPUT
    # =====================================================

    prediction_data = {

        "team1": team1,

        "team2": team2,

        "venue": venue,

        "toss_winner": toss_winner,

        "toss_decision": toss_decision,

        "team1_win_pct":
            team1_win_pct,

        "team2_win_pct":
            team2_win_pct,

        "team1_h2h":
            team1_h2h,

        "team2_h2h":
            team2_h2h,

        "team1_last5_wins":
            team1_last5_wins,

        "team2_last5_wins":
            team2_last5_wins,

        "team1_toss_win_pct":
            team1_toss_win_pct,

        "team2_toss_win_pct":
            team2_toss_win_pct,

        "team1_venue_win_pct":
            team1_venue_win_pct,

        "team2_venue_win_pct":
            team2_venue_win_pct,

        "toss_decision_success_rate":
            toss_decision_success_rate,

        "team1_batting_avg":
            team1_batting_avg,

        "team2_batting_avg":
            team2_batting_avg,

        "team1_bowling_economy":
            team1_bowling_economy,

        "team2_bowling_economy":
            team2_bowling_economy,

        "team1_powerplay_runs":
            team1_powerplay_runs,

        "team2_powerplay_runs":
            team2_powerplay_runs,

        "team1_death_runs":
            team1_death_runs,

        "team2_death_runs":
            team2_death_runs,

        "team1_boundary_pct":
            team1_boundary_pct,

        "team2_boundary_pct":
            team2_boundary_pct,

        "team1_run_rate":
            team1_run_rate,

        "team2_run_rate":
            team2_run_rate,

        "team1_wicket_ability":
            team1_wicket_ability,

        "team2_wicket_ability":
            team2_wicket_ability
    }


    X_input = pd.DataFrame(
        [prediction_data]
    )


    # =====================================================
    # FEATURE ORDER
    # =====================================================

    X_input = X_input[
        feature_columns
    ].copy()


    # =====================================================
    # ENCODING
    # =====================================================

    from sklearn.preprocessing import LabelEncoder


    categorical_columns = [
        "team1",
        "team2",
        "venue",
        "toss_winner",
        "toss_decision"
    ]


    for column in categorical_columns:

        encoder = LabelEncoder()

        encoder.fit(
            matches[column].astype(str)
        )

        X_input[column] = encoder.transform(
            X_input[column].astype(str)
        )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(
        X_input
    )


    winner = winner_encoder.inverse_transform(
        prediction.astype(int)
    )[0]


    # =====================================================
    # PROBABILITY
    # =====================================================

    winning_probability = None


    try:

        probabilities = model.predict_proba(
            X_input
        )


        predicted_class = prediction[0]


        class_index = np.where(
            model.classes_ == predicted_class
        )[0][0]


        winning_probability = (
            probabilities[0][class_index]
            * 100
        )


    except Exception:

        winning_probability = None


    # =====================================================
    # RESULT DISPLAY
    # =====================================================

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="result-label">'
        '🏆 Predicted Winner'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f'<div class="winner">{winner}</div>',
        unsafe_allow_html=True
    )


    if winning_probability is not None:

        st.markdown(
            f'<div class="probability">'
            f'📊 Model Probability: '
            f'{winning_probability:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )


    st.markdown(
        '<div style="margin-top:10px;">'
        'Based on the trained XGBoost Machine Learning model'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # MATCH DETAILS
    # =====================================================

    st.markdown(
        '<div class="details-box">',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="detail-title">'
        '🏏 Match Details'
        '</div>',
        unsafe_allow_html=True
    )


    detail_col1, detail_col2 = st.columns(2)


    with detail_col1:

        st.markdown(
            f"**Team 1:** {team1}"
        )

        st.markdown(
            f"**Team 2:** {team2}"
        )

        st.markdown(
            f"**Venue:** {venue}"
        )


    with detail_col2:

        st.markdown(
            f"**Toss Winner:** {toss_winner}"
        )

        st.markdown(
            f"**Toss Decision:** {toss_decision}"
        )

        st.markdown(
            "**Model:** XGBoost"
        )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # FOOTER
    # =====================================================

    st.caption(
        "Prediction is generated using the trained XGBoost "
        "model and engineered IPL historical features."
    )