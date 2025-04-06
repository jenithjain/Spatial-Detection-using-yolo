# Generated by Django 5.1.4 on 2025-04-06 00:27

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('staff', '0008_misplaceditemsanalysis_cleanliness_assessment_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelValidation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('validation_type', models.CharField(choices=[('object_detection', 'Object Detection'), ('gemini_comparison', 'Room Comparison'), ('damage_detection', 'Damage Detection')], max_length=30)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('name', models.CharField(blank=True, help_text='Optional name for this validation', max_length=100, null=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('checkin_image', models.ImageField(blank=True, null=True, upload_to='model_validation/checkin/')),
                ('checkout_image', models.ImageField(blank=True, null=True, upload_to='model_validation/checkout/')),
                ('checkin_annotated', models.ImageField(blank=True, null=True, upload_to='model_validation/checkin_annotated/')),
                ('checkout_annotated', models.ImageField(blank=True, null=True, upload_to='model_validation/checkout_annotated/')),
                ('checkin_objects', models.TextField(blank=True, help_text='JSON data of detected objects in checkin image', null=True)),
                ('checkout_objects', models.TextField(blank=True, help_text='JSON data of detected objects in checkout image', null=True)),
                ('missing_items', models.TextField(blank=True, help_text='JSON data of missing items', null=True)),
                ('gemini_analysis', models.TextField(blank=True, help_text='JSON data from Gemini room comparison', null=True)),
                ('analysis_image', models.ImageField(blank=True, null=True, upload_to='model_validation/comparison/')),
                ('damage_analysis', models.TextField(blank=True, help_text='JSON data from damage detection', null=True)),
                ('damage_image', models.ImageField(blank=True, null=True, upload_to='model_validation/damages/')),
                ('is_showcase', models.BooleanField(default=False, help_text='Whether this validation should be showcased')),
                ('showcase_order', models.IntegerField(default=0, help_text='Order to display in showcase')),
                ('staff_member', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='staff.staffmember')),
            ],
            options={
                'verbose_name_plural': 'Model Validations',
                'ordering': ['-created_at'],
            },
        ),
    ]
