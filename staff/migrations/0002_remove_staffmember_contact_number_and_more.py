# Generated by Django 5.1.4 on 2025-04-05 08:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('staff', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='staffmember',
            name='contact_number',
        ),
        migrations.RemoveField(
            model_name='staffmember',
            name='employee_id',
        ),
        migrations.AddField(
            model_name='staffmember',
            name='phone_number',
            field=models.CharField(default='', max_length=15),
        ),
        migrations.AddField(
            model_name='staffmember',
            name='role',
            field=models.CharField(default='Staff', max_length=50),
        ),
        migrations.AddField(
            model_name='staffmember',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
    ]
