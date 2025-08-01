
"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from DeTechProgress import app

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

