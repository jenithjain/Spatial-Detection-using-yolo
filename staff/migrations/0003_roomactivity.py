# Generated by Django 5.1.4 on 2025-04-05 11:49

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('staff', '0002_remove_staffmember_contact_number_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='RoomActivity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('room_number', models.CharField(max_length=20)),
                ('check_in_time', models.DateTimeField(auto_now_add=True)),
                ('check_out_time', models.DateTimeField(blank=True, null=True)),
                ('status', models.CharField(choices=[('active', 'Active'), ('completed', 'Completed')], default='active', max_length=10)),
                ('notes', models.TextField(blank=True, null=True)),
                ('yolo_session_id', models.CharField(blank=True, max_length=100, null=True)),
                ('has_missing_items', models.BooleanField(default=False)),
                ('missing_items_details', models.TextField(blank=True, null=True)),
                ('staff_member', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='staff.staffmember')),
            ],
            options={
                'verbose_name_plural': 'Room Activities',
                'ordering': ['-check_in_time'],
            },
        ),
    ]
