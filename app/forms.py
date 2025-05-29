from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, TextAreaField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo, URL, Optional
from app.models.user import User

class LoginForm(FlaskForm):
    """Form for user login"""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    """Form for user registration"""
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.find_by_username(username.data)
        if user is not None:
            raise ValidationError('Username already taken.')

    def validate_email(self, email):
        user = User.find_by_email(email.data)
        if user is not None:
            raise ValidationError('Email already registered.')

class DatasetForm(FlaskForm):
    """Form for dataset creation or editing"""
    title = StringField('Title', validators=[DataRequired(), Length(max=255)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=2000)])

    # File upload field
    dataset_file = FileField('Upload Dataset File', validators=[
        FileAllowed(['csv', 'json', 'xml', 'txt', 'tsv', 'xlsx', 'xls'],
                   'Only CSV, JSON, XML, TXT, TSV, and Excel files are allowed!')
    ])

    # Alternative URL field
    source_url = StringField('Or Dataset URL', validators=[Optional(), URL()])

    source = StringField('Source', validators=[Optional(), Length(max=128)])
    data_type = SelectField('Data Type', choices=[
        ('', 'Select Data Type'),
        ('tabular', 'Tabular Data'),
        ('text', 'Textual Data'),
        ('image', 'Image Data'),
        ('time_series', 'Time Series'),
        ('geo', 'Geospatial Data'),
        ('mixed', 'Mixed Data Types')
    ], validators=[Optional()])
    category = SelectField('Category', choices=[
        ('', 'Select Category'),
        ('education', 'Education'),
        ('health', 'Health'),
        ('agriculture', 'Agriculture'),
        ('environment', 'Environment'),
        ('social_science', 'Social Science'),
        ('economics', 'Economics'),
        ('other', 'Other')
    ], validators=[Optional()])
    tags = StringField('Tags (comma separated)', validators=[Optional(), Length(max=255)])
    submit = SubmitField('Submit')

class SearchForm(FlaskForm):
    """Form for dataset search"""
    query = StringField('Search', validators=[Optional()])
    category = SelectField('Category', choices=[
        ('', 'All Categories'),
        ('education', 'Education'),
        ('health', 'Health'),
        ('agriculture', 'Agriculture'),
        ('environment', 'Environment'),
        ('social_science', 'Social Science'),
        ('economics', 'Economics'),
        ('other', 'Other')
    ], validators=[Optional()])
    data_type = SelectField('Data Type', choices=[
        ('', 'All Types'),
        ('tabular', 'Tabular Data'),
        ('text', 'Textual Data'),
        ('image', 'Image Data'),
        ('time_series', 'Time Series'),
        ('geo', 'Geospatial Data'),
        ('mixed', 'Mixed Data Types')
    ], validators=[Optional()])
    submit = SubmitField('Search')

# Import ValidationError after it is used to avoid circular import
from wtforms.validators import ValidationError