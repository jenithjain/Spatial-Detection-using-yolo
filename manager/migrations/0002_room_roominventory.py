# Generated by Django 5.1.4 on 2025-04-05 14:53

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('manager', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Room',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('room_number', models.CharField(max_length=10)),
                ('room_type', models.CharField(default='Standard', max_length=50)),
                ('is_active', models.BooleanField(default=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('manager', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='manager.manager')),
            ],
            options={
                'unique_together': {('manager', 'room_number')},
            },
        ),
        migrations.CreateModel(
            name='RoomInventory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('item_name', models.CharField(max_length=100)),
                ('quantity', models.PositiveIntegerField(default=1)),
                ('description', models.TextField(blank=True, null=True)),
                ('added_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('room', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='inventory_items', to='manager.room')),
            ],
            options={
                'verbose_name_plural': 'Room Inventories',
            },
        ),
    ]
