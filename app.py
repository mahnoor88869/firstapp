"""
StudyBuddy - Your Smart Student Life
Streamlit UI + Groq LLM

Run:
    pip install -r requirements.txt
    export GROQ_API_KEY="your_key_here"     # or paste it in the sidebar
    streamlit run app.py
"""

import json
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="StudyBuddy", page_icon="🎓", layout="wide")

DATA_FILE = Path(__file__).with_name("studybuddy_data.json")
TODAY = date.today()
DEFAULT_MODEL = "llama-3.3-70b-versatile"

TASK_TYPES = ["Assignment", "Exam", "Quiz", "Project", "Lab", "Reading", "Other"]
PRIORITIES = ["High", "Medium", "Low"]
PRIORITY_W = {"High": 3.0, "Medium": 2.0, "Low": 1.0}
PRIORITY_XP = {"High": 20, "Medium": 10, "Low": 5}

# Percentage -> (letter, grade points). Edit to match your university.
GRADE_SCALE = [
    (90, "A+", 4.0), (85, "A", 4.0), (80, "A-", 3.67), (75, "B+", 3.33),
    (70, "B", 3.0), (65, "B-", 2.67), (60, "C+", 2.33), (55, "C", 2.0),
    (50, "D", 1.0), (0, "F", 0.0),
]
GRADE_POINTS = {g: p for _, g, p in GRADE_SCALE}
GRADE_CUTOFF = {g: c for c, g, _ in GRADE_SCALE}

# Virtual pet evolves with streak days
PET_STAGES = [
    (0, "Egg", "🥚"), (1, "Hatchling", "🐣"), (3, "Chick", "🐥"),
    (7, "Fox Cub", "🦊"), (14, "Wise Owl", "🦉"), (30, "Dragon", "🐲"),
]

REWARDS = [
    {"id": "rookie", "name": "Rookie Scholar", "cost": 50, "emoji": "🏅", "kind": "badge",
     "desc": "Your first badge. Everyone starts somewhere!"},
    {"id": "ai_boost", "name": "Motivation Boost", "cost": 100, "emoji": "⚡", "kind": "perk",
     "desc": "Unlocks an AI motivation button on your dashboard."},
    {"id": "cap", "name": "Graduation Cap", "cost": 200, "emoji": "🎓", "kind": "accessory",
     "desc": "A cap for your pet."},
    {"id": "glasses", "name": "Smart Glasses", "cost": 400, "emoji": "🕶️", "kind": "accessory",
     "desc": "Looks smart, studies smarter."},
    {"id": "crown", "name": "Golden Crown", "cost": 800, "emoji": "👑", "kind": "accessory",
     "desc": "For true study royalty."},
    {"id": "deans", "name": "Dean's List Badge", "cost": 1500, "emoji": "🏆", "kind": "badge",
     "desc": "Elite status."},
]

DEFAULT_DATA = {
    "tasks": [],
    "courses": [
        {"Course": "Course 1", "Credits": 3, "Grade": "A"},
        {"Course": "Course 2", "Credits": 3, "Grade": "B+"},
    ],
    "prev_cgpa": 0.0,
    "prev_credits": 0,
    "xp": 0,
    "streak": 0,
    "best_streak": 0,
    "last_active": None,
    "pet_name": "Buddy",
    "unlocked": [],
    "equipped": [],
    "completed_count": 0,
}

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
@keyframes bob {0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)}}
.pet {font-size:96px; text-align:center; animation: bob 2.2s ease-in-out infinite; line-height:1.1}
.pet-acc {text-align:center; font-size:28px; margin-top:-4px}
.pet-card {border:1px solid rgba(128,128,128,.3); border-radius:16px; padding:14px 10px;
           background:linear-gradient(135deg, rgba(99,102,241,.12), rgba(236,72,153,.10));}
.small-muted {opacity:.7; font-size:0.9rem}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_data() -> dict:
    data = json.loads(json.dumps(DEFAULT_DATA))
    if DATA_FILE.exists():
        try:
            data.update(json.loads(DATA_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    la = data.get("last_active")
    if la and date.fromisoformat(la) < TODAY - timedelta(days=1):
        data["streak"] = 0  # streak broken
    return data


def save_data():
    try:
        DATA_FILE.write_text(json.dumps(st.session_state.data, indent=2), encoding="utf-8")
    except Exception as e:
        st.warning(f"Could not save data: {e}")


if "data" not in st.session_state:
    st.session_state.data = load_data()
if "chat" not in st.session_state:
    st.session_state.chat = []


def D() -> dict:
    return st.session_state.data


# --------------------------------------------------------------------------- #
# Groq LLM helper
# --------------------------------------------------------------------------- #
def default_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    return key


def llm(messages, json_mode=False, temperature=0.4, max_tokens=1500) -> str:
    key = st.session_state.get("api_key", "")
    if not key:
        raise RuntimeError("Add your Groq API key in the sidebar first.")
    from groq import Groq

    client = Groq(api_key=key)
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=st.session_state.get("model", DEFAULT_MODEL),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content


def student_context() -> str:
    d = D()
    pending = [t for t in d["tasks"] if not t["done"]]
    pending.sort(key=lambda t: t["due"])
    lines = [
        f"- {t['title']} ({t['course']}, {t['type']}, due {t['due']}, {t['priority']} priority, ~{t['est_hours']}h)"
        for t in pending[:15]
    ] or ["- none"]
    return (
        f"Today: {TODAY.isoformat()}. Streak: {d['streak']} days. XP: {d['xp']}.\n"
        f"Previous CGPA: {d['prev_cgpa']} over {d['prev_credits']} credits.\n"
        "Pending tasks:\n" + "\n".join(lines)
    )


# --------------------------------------------------------------------------- #
# Game logic: pet, points, rewards
# --------------------------------------------------------------------------- #
def pet_stage(streak: int):
    stage = PET_STAGES[0]
    for s in PET_STAGES:
        if streak >= s[0]:
            stage = s
    idx = PET_STAGES.index(stage)
    nxt = PET_STAGES[idx + 1] if idx + 1 < len(PET_STAGES) else None
    return stage, nxt


def render_pet(compact=False):
    d = D()
    (_, name, emoji), nxt = pet_stage(d["streak"])
    acc = "".join(r["emoji"] for r in REWARDS if r["id"] in d["equipped"])
    active_today = d["last_active"] == TODAY.isoformat()
    if active_today:
        mood = "😄 Happy and fed!"
    elif d["streak"] > 0:
        mood = "😟 Hungry! Finish a task today to keep your streak."
    else:
        mood = "😴 Sleepy. Complete a task to wake me up!"
    st.markdown(
        f"<div class='pet-card'><div class='pet'>{emoji}</div>"
        f"<div class='pet-acc'>{acc}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{d['pet_name']}** the {name}")
    st.caption(mood)
    if nxt and not compact:
        need = nxt[0] - d["streak"]
        st.progress(min(d["streak"] / nxt[0], 1.0), text=f"{need} more streak day(s) to become a {nxt[1]} {nxt[2]}")
    elif not nxt and not compact:
        st.success("Max evolution reached! 🐲")


def calc_points(task: dict) -> int:
    d = D()
    pts = 10 + PRIORITY_XP.get(task["priority"], 5) + int(float(task["est_hours"]) * 5)
    pts += min(d["streak"], 10) * 2
    if date.fromisoformat(task["due"]) >= TODAY:
        pts += 5  # on-time bonus
    return pts


def complete_task(task: dict) -> int:
    d = D()
    task["done"] = True
    task["done_on"] = TODAY.isoformat()
    la = d["last_active"]
    if la != TODAY.isoformat():
        if la and date.fromisoformat(la) == TODAY - timedelta(days=1):
            d["streak"] += 1
        else:
            d["streak"] = 1
        d["last_active"] = TODAY.isoformat()
    d["best_streak"] = max(d["best_streak"], d["streak"])
    pts = calc_points(task)
    d["xp"] += pts
    d["completed_count"] += 1
    for r in REWARDS:
        if d["xp"] >= r["cost"] and r["id"] not in d["unlocked"]:
            d["unlocked"].append(r["id"])
            st.toast(f"Reward unlocked: {r['emoji']} {r['name']}!", icon="🎉")
    save_data()
    return pts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def days_left(task: dict) -> int:
    return (date.fromisoformat(task["due"]) - TODAY).days


def due_badge(task: dict) -> str:
    dl = days_left(task)
    if dl < 0:
        return f"🔴 Overdue by {-dl}d"
    if dl == 0:
        return "🟠 Due today"
    if dl <= 3:
        return f"🟠 {dl}d left"
    return f"🟢 {dl}d left"


def pending_tasks():
    return sorted([t for t in D()["tasks"] if not t["done"]], key=lambda t: (t["due"], -PRIORITY_W[t["priority"]]))


def add_task(title, course, ttype, due, priority, hours):
    D()["tasks"].append(
        {
            "id": uuid.uuid4().hex[:8],
            "title": title.strip(),
            "course": course.strip() or "General",
            "type": ttype,
            "due": due.isoformat() if isinstance(due, (date, datetime)) else str(due),
            "priority": priority,
            "est_hours": float(hours),
            "done": False,
            "done_on": None,
        }
    )


def percent_to_grade(pct: float):
    for cutoff, letter, pts in GRADE_SCALE:
        if pct >= cutoff:
            return letter, pts
    return "F", 0.0


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎓 StudyBuddy")
    st.caption("Your smart student life, without the overwhelm.")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📚 Tasks & Deadlines", "🗓️ Daily Study Plan",
         "🎯 CGPA & Marks", "🎁 Rewards", "🤖 AI Assistant"],
        label_visibility="collapsed",
    )
    st.divider()
    render_pet(compact=True)
    c1, c2 = st.columns(2)
    c1.metric("🔥 Streak", D()["streak"])
    c2.metric("⭐ XP", D()["xp"])
    st.divider()
    with st.expander("⚙️ Settings"):
        st.session_state.api_key = st.text_input(
            "Groq API key", value=default_key(), type="password",
            help="Tip: set the GROQ_API_KEY environment variable instead of pasting it here.",
        )
        st.session_state.model = st.text_input("Groq model", value=DEFAULT_MODEL)
        new_name = st.text_input("Pet name", value=D()["pet_name"])
        if new_name != D()["pet_name"]:
            D()["pet_name"] = new_name or "Buddy"
            save_data()
        if st.button("Reset all data"):
            st.session_state.data = json.loads(json.dumps(DEFAULT_DATA))
            save_data()
            st.rerun()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard():
    d = D()
    st.header(f"Hey there! 👋  {TODAY.strftime('%A, %d %B %Y')}")
    left, right = st.columns([1, 2])
    with left:
        render_pet()
    with right:
        pend = pending_tasks()
        overdue = [t for t in pend if days_left(t) < 0]
        week = [t for t in pend if 0 <= days_left(t) <= 7]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pending", len(pend))
        m2.metric("Due in 7 days", len(week))
        m3.metric("Overdue", len(overdue))
        m4.metric("Completed", d["completed_count"])
        st.subheader("Next up")
        if not pend:
            st.info("No pending tasks. Add some in **Tasks & Deadlines**.")
        for t in pend[:5]:
            st.write(f"**{t['title']}** · {t['course']} · {t['type']} · {due_badge(t)} · {t['priority']}")

    st.subheader("Workload for the next 14 days (estimated hours by due date)")
    days = [TODAY + timedelta(days=i) for i in range(14)]
    load = {x.strftime("%d %b"): 0.0 for x in days}
    for t in pending_tasks():
        due = date.fromisoformat(t["due"])
        key = (due if due >= TODAY else TODAY).strftime("%d %b")
        if key in load:
            load[key] += t["est_hours"]
    st.bar_chart(pd.DataFrame({"Hours": load}))

    if "ai_boost" in d["unlocked"]:
        if st.button("⚡ Give me a motivation boost"):
            try:
                with st.spinner("Thinking..."):
                    msg = llm(
                        [
                            {"role": "system", "content": "You are a warm, upbeat study coach. Keep replies under 80 words."},
                            {"role": "user", "content": student_context() + "\nGive me a short personalized pep talk."},
                        ],
                        temperature=0.8,
                    )
                st.success(msg)
            except Exception as e:
                st.error(str(e))


def page_tasks():
    st.header("📚 Tasks & Deadlines")

    with st.expander("➕ Add a task", expanded=not D()["tasks"]):
        with st.form("add_task", clear_on_submit=True):
            c1, c2 = st.columns(2)
            title = c1.text_input("Title*")
            course = c2.text_input("Course")
            c3, c4, c5, c6 = st.columns(4)
            ttype = c3.selectbox("Type", TASK_TYPES)
            due = c4.date_input("Due date", value=TODAY + timedelta(days=3))
            priority = c5.selectbox("Priority", PRIORITIES, index=1)
            hours = c6.number_input("Est. hours", 0.5, 100.0, 2.0, 0.5)
            if st.form_submit_button("Add task"):
                if title.strip():
                    add_task(title, course, ttype, due, priority, hours)
                    save_data()
                    st.success("Task added!")
                    st.rerun()
                else:
                    st.error("Please enter a title.")

    with st.expander("✨ Smart add: paste a syllabus / message and let AI extract deadlines"):
        text = st.text_area("Paste text (e.g. 'Math assignment 2 due next Friday, physics midterm on 12 Dec...')", height=140)
        if st.button("Extract tasks with AI"):
            if not text.strip():
                st.warning("Paste some text first.")
            else:
                try:
                    with st.spinner("Reading..."):
                        sys_prompt = (
                            "You extract university tasks from text. Respond ONLY with a JSON object: "
                            '{"tasks":[{"title":str,"course":str,"type":one of '
                            f"{TASK_TYPES},"
                            '"due":"YYYY-MM-DD","priority":"High|Medium|Low","est_hours":number}]}. '
                            f"Today is {TODAY.isoformat()} ({TODAY.strftime('%A')}). Resolve relative dates. "
                            "Exams and projects are High priority unless stated otherwise."
                        )
                        raw = llm(
                            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
                            json_mode=True, temperature=0.1,
                        )
                        items = json.loads(raw).get("tasks", [])
                    rows = []
                    for it in items:
                        try:
                            rows.append(
                                {
                                    "Add": True,
                                    "title": str(it.get("title", "Untitled")),
                                    "course": str(it.get("course", "General")),
                                    "type": it.get("type") if it.get("type") in TASK_TYPES else "Other",
                                    "due": pd.to_datetime(it.get("due")).date(),
                                    "priority": it.get("priority") if it.get("priority") in PRIORITIES else "Medium",
                                    "est_hours": float(it.get("est_hours", 2)),
                                }
                            )
                        except Exception:
                            continue
                    st.session_state.parsed_df = pd.DataFrame(rows)
                    if not rows:
                        st.warning("No tasks found in that text.")
                except Exception as e:
                    st.error(f"AI error: {e}")

        if "parsed_df" in st.session_state and not st.session_state.parsed_df.empty:
            st.write("Review, edit, then add:")
            edited = st.data_editor(
                st.session_state.parsed_df,
                column_config={
                    "type": st.column_config.SelectboxColumn(options=TASK_TYPES),
                    "priority": st.column_config.SelectboxColumn(options=PRIORITIES),
                    "due": st.column_config.DateColumn(),
                },
                hide_index=True, use_container_width=True, key="parsed_editor",
            )
            if st.button("Add selected tasks"):
                n = 0
                for _, r in edited.iterrows():
                    if r["Add"]:
                        add_task(r["title"], r["course"], r["type"], r["due"], r["priority"], r["est_hours"])
                        n += 1
                save_data()
                del st.session_state["parsed_df"]
                st.success(f"Added {n} task(s).")
                st.rerun()

    st.subheader("Pending")
    pend = pending_tasks()
    if not pend:
        st.info("Nothing pending. 🎉")
    for t in pend:
        c = st.columns([4, 2, 1.3, 1.1, 1.1, 0.8])
        c[0].markdown(f"**{t['title']}**  \n<span class='small-muted'>{t['course']} · {t['type']}</span>", unsafe_allow_html=True)
        c[1].write(f"{due_badge(t)}  \n{t['due']}")
        c[2].write(t["priority"])
        c[3].write(f"{t['est_hours']}h")
        if c[4].button("✅ Done", key=f"done_{t['id']}"):
            pts = complete_task(t)
            st.toast(f"+{pts} XP! Streak: {D()['streak']} 🔥", icon="⭐")
            st.rerun()
        if c[5].button("🗑️", key=f"del_{t['id']}"):
            D()["tasks"] = [x for x in D()["tasks"] if x["id"] != t["id"]]
            save_data()
            st.rerun()

    done = [t for t in D()["tasks"] if t["done"]]
    with st.expander(f"Completed ({len(done)})"):
        for t in sorted(done, key=lambda x: x.get("done_on") or "", reverse=True):
            st.write(f"✔️ {t['title']} · {t['course']} · done {t.get('done_on')}")


def build_plan(hours: float, start: datetime):
    pend = pending_tasks()
    scored = []
    for t in pend:
        dl = max(days_left(t), 0)
        urg = PRIORITY_W[t["priority"]] * (1 + 7 / (dl + 1))
        if t["type"] in ("Exam", "Quiz"):
            urg *= 1.2
        need = t["est_hours"] if dl <= 1 else max(min(0.5, t["est_hours"]), t["est_hours"] / dl)
        scored.append((urg, t, need))
    scored.sort(key=lambda x: -x[0])

    remaining = hours
    alloc = {}
    for _, t, need in scored:
        if remaining <= 0.01:
            break
        a = min(need, remaining)
        alloc[t["id"]] = a
        remaining -= a
    # use leftover time on the most urgent tasks
    for _, t, _ in scored:
        if remaining <= 0.01:
            break
        if t["id"] in alloc:
            extra = min(t["est_hours"] - alloc[t["id"]], remaining)
            if extra > 0:
                alloc[t["id"]] += extra
                remaining -= extra

    rows, cur = [], start
    for _, t, _ in scored:
        if t["id"] not in alloc:
            continue
        mins = max(15, int(round(alloc[t["id"]] * 60 / 5) * 5))
        end = cur + timedelta(minutes=mins)
        dl = days_left(t)
        why = "Overdue!" if dl < 0 else "Due today" if dl == 0 else f"Due in {dl}d"
        rows.append(
            {"Time": f"{cur:%H:%M} - {end:%H:%M}", "Task": t["title"], "Course": t["course"],
             "Minutes": mins, "Why": f"{why} · {t['priority']}"}
        )
        cur = end + timedelta(minutes=10)  # 10 min break
    return rows, max(remaining, 0)


def page_plan():
    st.header("🗓️ Daily Study Plan")
    c1, c2 = st.columns(2)
    hours = c1.slider("Hours available today", 0.5, 12.0, 3.0, 0.5)
    start_t = c2.time_input("Start time", value=datetime.strptime("17:00", "%H:%M").time())
    if not pending_tasks():
        st.info("Add tasks first and I'll build your plan.")
        return
    rows, left = build_plan(hours, datetime.combine(TODAY, start_t))
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption("10-minute breaks are added between blocks. Tasks are ranked by priority and how soon they're due.")
        if left > 0.25:
            st.success(f"You have ~{left:.1f}h of spare time. Nice! Use it for revision or rest.")
    else:
        st.info("No time allocated.")

    if rows and st.button("🤖 Get AI coaching for this plan"):
        try:
            with st.spinner("Coaching..."):
                msg = llm(
                    [
                        {"role": "system", "content": "You are a friendly study coach. Give 4-5 concise, practical tips (study techniques, focus, break use) tailored to this plan. Use short bullets."},
                        {"role": "user", "content": student_context() + "\n\nToday's plan:\n" + pd.DataFrame(rows).to_string(index=False)},
                    ],
                    temperature=0.6,
                )
            st.markdown(msg)
        except Exception as e:
            st.error(str(e))


def page_cgpa():
    d = D()
    st.header("🎯 CGPA & Required Marks")
    t1, t2, t3 = st.tabs(["CGPA calculator", "Target planner", "Required marks in a course"])

    with t1:
        c1, c2 = st.columns(2)
        d["prev_credits"] = c1.number_input("Previously completed credits", 0, 300, int(d["prev_credits"]))
        d["prev_cgpa"] = c2.number_input("Previous CGPA (4.0 scale)", 0.0, 4.0, float(d["prev_cgpa"]), 0.01)
        st.write("Current semester courses")
        df = pd.DataFrame(d["courses"])
        edited = st.data_editor(
            df, num_rows="dynamic", hide_index=True, use_container_width=True,
            column_config={
                "Credits": st.column_config.NumberColumn(min_value=0.5, max_value=10, step=0.5),
                "Grade": st.column_config.SelectboxColumn(options=list(GRADE_POINTS.keys())),
            },
            key="courses_editor",
        )
        clean = edited.dropna(subset=["Credits", "Grade"])
        d["courses"] = clean.to_dict("records")
        save_data()
        sem_cr = float(clean["Credits"].sum()) if not clean.empty else 0.0
        sem_pts = float(sum(r["Credits"] * GRADE_POINTS[r["Grade"]] for _, r in clean.iterrows())) if not clean.empty else 0.0
        sem_gpa = sem_pts / sem_cr if sem_cr else 0.0
        tot_cr = d["prev_credits"] + sem_cr
        tot_pts = d["prev_cgpa"] * d["prev_credits"] + sem_pts
        cgpa = tot_pts / tot_cr if tot_cr else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("Semester GPA", f"{sem_gpa:.2f}")
        m2.metric("Updated CGPA", f"{cgpa:.2f}")
        m3.metric("Total credits", f"{tot_cr:g}")

    with t2:
        st.write("What GPA do I need in my remaining credits to reach a target CGPA?")
        c1, c2 = st.columns(2)
        target = c1.number_input("Target CGPA", 0.0, 4.0, 3.5, 0.01)
        remaining = c2.number_input("Credits remaining after this semester", 1, 200, 60)
        req = (target * (tot_cr + remaining) - tot_pts) / remaining
        st.metric("Required average GPA in remaining credits", f"{req:.2f}")
        if req > 4.0:
            st.error("Not reachable with the remaining credits. Consider a lower target or retaking courses.")
        elif req <= 0:
            st.success("You've already secured this target!")
        else:
            letter = next((g for _, g, p in sorted(GRADE_SCALE, key=lambda x: x[2]) if p >= req), "A")
            st.info(f"That is roughly a **{letter}** average (≈ {GRADE_CUTOFF[letter]}%+) across your remaining courses.")

    with t3:
        st.write("Enter your assessment weights and scores. Leave **Score %** empty for pending ones.")
        base = pd.DataFrame(
            [
                {"Component": "Assignments", "Weight %": 20.0, "Score %": 85.0},
                {"Component": "Quizzes", "Weight %": 10.0, "Score %": 70.0},
                {"Component": "Midterm", "Weight %": 30.0, "Score %": None},
                {"Component": "Final", "Weight %": 40.0, "Score %": None},
            ]
        )
        comp = st.data_editor(
            base, num_rows="dynamic", hide_index=True, use_container_width=True, key="marks_editor",
            column_config={
                "Weight %": st.column_config.NumberColumn(min_value=0, max_value=100),
                "Score %": st.column_config.NumberColumn(min_value=0, max_value=100),
            },
        )
        grade_target = st.selectbox("Target grade", [g for _, g, _ in GRADE_SCALE if g != "F"], index=2)
        target_pct = GRADE_CUTOFF[grade_target]
        w = pd.to_numeric(comp["Weight %"], errors="coerce").fillna(0)
        s = pd.to_numeric(comp["Score %"], errors="coerce")
        done_mask = s.notna()
        earned = float((w[done_mask] * s[done_mask] / 100).sum())
        done_w = float(w[done_mask].sum())
        rem_w = float(w[~done_mask].sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Marks earned so far", f"{earned:.1f} / {done_w:.0f}")
        k2.metric("Weight remaining", f"{rem_w:.0f}%")
        k3.metric("Target", f"{target_pct}% ({grade_target})")
        if rem_w <= 0:
            letter, _ = percent_to_grade(earned)
            st.info(f"All components are graded. Final: **{earned:.1f}%** → **{letter}**")
        else:
            need = (target_pct - earned) / rem_w * 100
            if need > 100:
                st.error(f"You would need {need:.1f}% on the remaining work. Not reachable for {grade_target}. Try a lower target.")
            elif need <= 0:
                st.success(f"You've already secured {grade_target}! 🎉")
            else:
                st.success(f"You need an average of **{need:.1f}%** on the remaining {rem_w:.0f}% of the course to get **{grade_target}**.")
        if abs(float(w.sum()) - 100) > 0.01:
            st.warning(f"Weights add up to {w.sum():g}%, not 100%.")


def page_rewards():
    d = D()
    st.header("🎁 Rewards")
    level = d["xp"] // 100 + 1
    c1, c2, c3 = st.columns(3)
    c1.metric("Level", level)
    c2.metric("Total XP", d["xp"])
    c3.metric("Best streak", f"{d['best_streak']} 🔥")
    st.progress((d["xp"] % 100) / 100, text=f"{100 - d['xp'] % 100} XP to level {level + 1}")
    st.caption("Earn XP by completing tasks: base 10 + priority + effort + streak bonus + on-time bonus.")

    for r in REWARDS:
        unlocked = r["id"] in d["unlocked"] or d["xp"] >= r["cost"]
        c = st.columns([0.6, 3, 1.5, 1.5])
        c[0].markdown(f"## {r['emoji']}")
        c[1].markdown(f"**{r['name']}** ({r['kind']})  \n<span class='small-muted'>{r['desc']}</span>", unsafe_allow_html=True)
        if unlocked:
            c[2].success("Unlocked")
            if r["kind"] == "accessory":
                on = r["id"] in d["equipped"]
                if c[3].toggle("Equip", value=on, key=f"eq_{r['id']}") != on:
                    if on:
                        d["equipped"].remove(r["id"])
                    else:
                        d["equipped"].append(r["id"])
                    save_data()
                    st.rerun()
        else:
            c[2].write(f"🔒 {r['cost']} XP")
            c[3].progress(min(d["xp"] / r["cost"], 1.0))

    st.subheader("Pet evolution path")
    st.write("  →  ".join(f"{e} {n} ({s}d)" for s, n, e in PET_STAGES))


def page_assistant():
    st.header("🤖 AI Study Assistant")
    st.caption("Ask for study tips, help prioritizing, exam strategies, or how to catch up on a backlog.")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if prompt := st.chat_input("Ask StudyBuddy anything..."):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                system = (
                    "You are StudyBuddy, a supportive, practical university study coach. "
                    "Be concise and actionable. Reduce overwhelm: break things into small steps.\n\n"
                    + student_context()
                )
                msgs = [{"role": "system", "content": system}] + st.session_state.chat[-12:]
                with st.spinner("Thinking..."):
                    reply = llm(msgs, temperature=0.6)
            except Exception as e:
                reply = f"⚠️ {e}"
            st.markdown(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
{
    "🏠 Dashboard": page_dashboard,
    "📚 Tasks & Deadlines": page_tasks,
    "🗓️ Daily Study Plan": page_plan,
    "🎯 CGPA & Marks": page_cgpa,
    "🎁 Rewards": page_rewards,
    "🤖 AI Assistant": page_assistant,
}[page]()
