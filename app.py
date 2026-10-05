# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.authenticated:

    st.markdown(
        """
        <style>

        .login-box {
            max-width: 460px;
            margin: 70px auto 20px auto;
            padding: 35px;
            background: #141722;
            border: 1px solid rgba(192,132,252,0.25);
            border-radius: 22px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # Professional heading
    st.markdown(
        "<h2 style='text-align:center;'>🩸 SICKLESCAN</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; color:#a1a1aa;'>"
        "AI-Powered Blood Smear Analysis"
        "</p>",
        unsafe_allow_html=True
    )

    st.write("")

    login_tab, signup_tab = st.tabs(
        ["🔐 Login", "✨ Create Account"]
    )

    # =====================================================
    # LOGIN
    # =====================================================

    with login_tab:

        st.subheader("Welcome back")
        st.caption("Sign in to continue to SickleScan.")

        with st.form("login_form"):

            email = st.text_input(
                "Email",
                placeholder="you@example.com"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )

            submitted = st.form_submit_button(
                "Login",
                use_container_width=True
            )

        if submitted:

            if not email or not password:

                st.error(
                    "Please enter your email and password."
                )

            else:

                try:

                    response = login_user(
                        email.strip(),
                        password
                    )

                    if response.ok:

                        data = response.json()

                        st.session_state.authenticated = True
                        st.session_state.user_email = email.strip()
                        st.session_state.access_token = data.get(
                            "access_token"
                        )
                        st.session_state.refresh_token = data.get(
                            "refresh_token"
                        )

                        st.rerun()

                    else:

                        payload = response.json()

                        message = (
                            payload.get("msg")
                            or payload.get("error_description")
                            or "Invalid email or password."
                        )

                        st.error(message)

                except requests.RequestException:

                    st.error(
                        "Unable to connect to authentication service."
                    )

    # =====================================================
    # SIGN UP
    # =====================================================

    with signup_tab:

        st.subheader("Create your account")
        st.caption("Create an account to use SickleScan.")

        with st.form("signup_form"):

            new_email = st.text_input(
                "Email",
                placeholder="you@example.com",
                key="signup_email"
            )

            new_password = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 6 characters",
                key="signup_password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter your password",
                key="confirm_password"
            )

            signup_submitted = st.form_submit_button(
                "Create Account",
                use_container_width=True
            )

        if signup_submitted:

            if not new_email or not new_password or not confirm_password:

                st.error(
                    "Please fill in all fields."
                )

            elif len(new_password) < 6:

                st.error(
                    "Password must be at least 6 characters."
                )

            elif new_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                try:

                    response = signup_user(
                        new_email.strip(),
                        new_password
                    )

                    if response.ok:

                        data = response.json()

                        if data.get("access_token"):

                            st.session_state.authenticated = True
                            st.session_state.user_email = new_email.strip()
                            st.session_state.access_token = data.get(
                                "access_token"
                            )
                            st.session_state.refresh_token = data.get(
                                "refresh_token"
                            )

                            st.rerun()

                        else:

                            st.success(
                                "Account created! "
                                "Check your email to confirm your account."
                            )

                    else:

                        payload = response.json()

                        message = (
                            payload.get("msg")
                            or payload.get("error_description")
                            or "Could not create the account."
                        )

                        st.error(message)

                except requests.RequestException:

                    st.error(
                        "Unable to connect to authentication service."
                    )

    st.stop()
