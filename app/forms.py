from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, TextAreaField, PasswordField, BooleanField, SubmitField, SelectField, RadioField
from wtforms.validators import DataRequired, Email, Length, EqualTo, URL, Optional, NumberRange
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
    """Form for dataset creation or editing with enhanced automatic field extraction"""
    title = StringField('Title (auto-generated if not provided)', validators=[Optional(), Length(max=255)])
    description = TextAreaField('Description (auto-generated if not provided)', validators=[Optional(), Length(max=2000)])

    # File upload field
    dataset_file = FileField('Upload Dataset File', validators=[
        FileAllowed(['csv', 'json', 'xml', 'txt', 'tsv', 'xlsx', 'xls', 'zip'],
                   'Only CSV, JSON, XML, TXT, TSV, Excel, and ZIP files are allowed!')
    ])

    # Alternative URL field
    source_url = StringField('Or Dataset URL', validators=[Optional(), URL()])

    source = StringField('Source (auto-detected if not provided)', validators=[Optional(), Length(max=128)])
    data_type = SelectField('Data Type (auto-detected if not provided)', choices=[
        ('', 'Auto-detect Data Type'),
        ('tabular', 'Tabular Data'),
        ('text', 'Textual Data'),
        ('image', 'Image Data'),
        ('time_series', 'Time Series'),
        ('geo', 'Geospatial Data'),
        ('mixed', 'Mixed Data Types'),
        ('collection', 'Dataset Collection')
    ], validators=[Optional()])
    category = SelectField('Category (auto-detected if not provided)', choices=[
        ('', 'Auto-detect Category'),
        ('education', 'Education'),
        ('health', 'Health'),
        ('agriculture', 'Agriculture'),
        ('environment', 'Environment'),
        ('social_science', 'Social Science'),
        ('economics', 'Economics'),
        ('finance', 'Finance'),
        ('retail', 'Retail'),
        ('technology', 'Technology'),
        ('government', 'Government'),
        ('research', 'Research'),
        ('other', 'Other')
    ], validators=[Optional()])
    tags = StringField('Tags (auto-generated if not provided)', validators=[Optional(), Length(max=500)])

    # Additional optional fields for better metadata
    license = StringField('License (auto-detected if not provided)', validators=[Optional(), Length(max=255)])
    author = StringField('Author/Publisher (defaults to uploader)', validators=[Optional(), Length(max=255)])

    submit = SubmitField('Upload & Process Dataset')

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


class FixQueueForm(FlaskForm):
    """Simple form for fixing stuck queue items"""
    submit = SubmitField('Fix Stuck Items')

class FeedbackForm(FlaskForm):
    rating = RadioField('Overall Rating', choices=[('1', '1 Star'), ('2', '2 Stars'), ('3', '3 Stars'), ('4', '4 Stars'), ('5', '5 Stars')], validators=[DataRequired()])
    satisfaction = SelectField('Satisfaction', choices=[('', 'Select...'), ('1', 'Very Dissatisfied'), ('2', 'Dissatisfied'), ('3', 'Neutral'), ('4', 'Satisfied'), ('5', 'Very Satisfied')], validators=[Optional()])
    usefulness = SelectField('Usefulness', choices=[('', 'Select...'), ('1', 'Not Useful'), ('2', 'Slightly Useful'), ('3', 'Moderately Useful'), ('4', 'Very Useful'), ('5', 'Extremely Useful')], validators=[Optional()])
    quality = SelectField('Data Quality', choices=[('', 'Select...'), ('1', 'Poor'), ('2', 'Fair'), ('3', 'Good'), ('4', 'Very Good'), ('5', 'Excellent')], validators=[Optional()])
    comment = TextAreaField('Your Review', validators=[Optional(), Length(max=2000)])
    feedback_type = RadioField('Feedback Type', choices=[('rating', 'Rating'), ('comment', 'Review'), ('suggestion', 'Suggestion'), ('issue', 'Issue')], default='rating', validators=[DataRequired()])
    is_anonymous = BooleanField('Submit anonymously')
    submit = SubmitField('Submit Feedback')
