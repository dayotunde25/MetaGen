from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from urllib.parse import urlparse

from app.models.user import User
from app.forms import LoginForm, RegistrationForm

# Create blueprint
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = LoginForm()

    if form.validate_on_submit():
        user = User.find_by_username(form.username.data)

        if user is None or not user.verify_password(form.password.data):
            flash('Invalid username or password', 'error')
            return redirect(url_for('auth.login'))

        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')

        # Validate next parameter to avoid open redirect vulnerability
        if not next_page or urlparse(next_page).netloc != '':
            next_page = url_for('main.dashboard')

        return redirect(next_page)

    return render_template('auth/login.html', title='Sign In', form=form)

@auth_bp.route('/logout')
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RegistrationForm()

    if form.validate_on_submit():
        user = User.create(
            username=form.username.data,
            email=form.email.data,
            password=form.password.data
        )

        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html', title='Register', form=form)