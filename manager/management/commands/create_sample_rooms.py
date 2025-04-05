from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from manager.models import Manager, Room, RoomInventory

class Command(BaseCommand):
    help = 'Creates sample rooms with inventory items for demonstration'

    def add_arguments(self, parser):
        parser.add_argument('--manager_username', type=str, default=None, help='Username of the manager to associate rooms with')

    def handle(self, *args, **options):
        manager_username = options['manager_username']

        # Get the first manager if username not provided
        if manager_username:
            try:
                manager = Manager.objects.get(user__username=manager_username)
            except Manager.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'Manager with username {manager_username} not found!'))
                return
        else:
            try:
                manager = Manager.objects.first()
                if not manager:
                    self.stdout.write(self.style.ERROR('No managers found in the system. Please create a manager first.'))
                    return
            except Manager.DoesNotExist:
                self.stdout.write(self.style.ERROR('No managers found in the system. Please create a manager first.'))
                return

        # Room data with inventory
        room_data = [
            {
                'number': '101',
                'type': 'Standard',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'Queen size bed'},
                    {'name': 'Chair', 'quantity': 2, 'description': 'Wooden chairs'},
                    {'name': 'TV', 'quantity': 1, 'description': '32-inch LCD TV'},
                    {'name': 'Table', 'quantity': 1, 'description': 'Writing desk'},
                    {'name': 'Lamp', 'quantity': 2, 'description': 'Bedside lamps'}
                ]
            },
            {
                'number': '102',
                'type': 'Deluxe',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'King size bed'},
                    {'name': 'Chair', 'quantity': 2, 'description': 'Upholstered chairs'},
                    {'name': 'TV', 'quantity': 1, 'description': '42-inch Smart TV'},
                    {'name': 'Table', 'quantity': 1, 'description': 'Work desk'},
                    {'name': 'Lamp', 'quantity': 2, 'description': 'Designer lamps'},
                    {'name': 'Sofa', 'quantity': 1, 'description': 'Two-seater sofa'}
                ]
            },
            {
                'number': '103',
                'type': 'Standard',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'Queen size bed'},
                    {'name': 'Chair', 'quantity': 1, 'description': 'Wooden chair'},
                    {'name': 'TV', 'quantity': 1, 'description': '32-inch LCD TV'},
                    {'name': 'Table', 'quantity': 1, 'description': 'Small table'},
                    {'name': 'Lamp', 'quantity': 1, 'description': 'Desk lamp'}
                ]
            },
            {
                'number': '104',
                'type': 'Suite',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'King size bed'},
                    {'name': 'Chair', 'quantity': 4, 'description': 'Luxury chairs'},
                    {'name': 'TV', 'quantity': 2, 'description': '50-inch Smart TVs'},
                    {'name': 'Table', 'quantity': 2, 'description': 'Dining and work tables'},
                    {'name': 'Lamp', 'quantity': 3, 'description': 'Designer lamps'},
                    {'name': 'Sofa', 'quantity': 1, 'description': 'Sectional sofa'},
                    {'name': 'Minibar', 'quantity': 1, 'description': 'Stocked minibar'}
                ]
            },
            {
                'number': '105',
                'type': 'Standard',
                'inventory': [
                    {'name': 'Bed', 'quantity': 2, 'description': 'Twin beds'},
                    {'name': 'Chair', 'quantity': 2, 'description': 'Wooden chairs'},
                    {'name': 'TV', 'quantity': 1, 'description': '32-inch LCD TV'},
                    {'name': 'Table', 'quantity': 1, 'description': 'Writing desk'},
                    {'name': 'Lamp', 'quantity': 2, 'description': 'Bedside lamps'}
                ]
            },
            {
                'number': '106',
                'type': 'Deluxe',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'King size bed'},
                    {'name': 'Chair', 'quantity': 3, 'description': 'Upholstered chairs'},
                    {'name': 'TV', 'quantity': 1, 'description': '42-inch Smart TV'},
                    {'name': 'Table', 'quantity': 1, 'description': 'Work desk'},
                    {'name': 'Lamp', 'quantity': 2, 'description': 'Designer lamps'},
                    {'name': 'Coffee Machine', 'quantity': 1, 'description': 'Espresso machine'}
                ]
            },
            {
                'number': '107',
                'type': 'Family',
                'inventory': [
                    {'name': 'Bed', 'quantity': 2, 'description': 'Queen size beds'},
                    {'name': 'Chair', 'quantity': 4, 'description': 'Comfortable chairs'},
                    {'name': 'TV', 'quantity': 1, 'description': '42-inch Smart TV'},
                    {'name': 'Table', 'quantity': 2, 'description': 'Work and dining tables'},
                    {'name': 'Lamp', 'quantity': 3, 'description': 'Various lamps'},
                    {'name': 'Sofa', 'quantity': 1, 'description': 'Pull-out sofa bed'}
                ]
            },
            {
                'number': '108',
                'type': 'Standard',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'Queen size bed'},
                    {'name': 'Chair', 'quantity': 2, 'description': 'Wooden chairs'},
                    {'name': 'TV', 'quantity': 1, 'description': '32-inch LCD TV'},
                    {'name': 'Table', 'quantity': 1, 'description': 'Writing desk'},
                    {'name': 'Lamp', 'quantity': 2, 'description': 'Bedside lamps'}
                ]
            },
            {
                'number': '109',
                'type': 'Deluxe',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'King size bed'},
                    {'name': 'Chair', 'quantity': 2, 'description': 'Upholstered chairs'},
                    {'name': 'TV', 'quantity': 1, 'description': '42-inch Smart TV'},
                    {'name': 'Table', 'quantity': 1, 'description': 'Work desk'},
                    {'name': 'Lamp', 'quantity': 2, 'description': 'Designer lamps'},
                    {'name': 'Fridge', 'quantity': 1, 'description': 'Mini refrigerator'}
                ]
            },
            {
                'number': '110',
                'type': 'Presidential Suite',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'California King bed'},
                    {'name': 'Chair', 'quantity': 6, 'description': 'Luxury chairs'},
                    {'name': 'TV', 'quantity': 3, 'description': '55-inch OLED TVs'},
                    {'name': 'Table', 'quantity': 3, 'description': 'Dining, work, and coffee tables'},
                    {'name': 'Lamp', 'quantity': 5, 'description': 'Designer lamps'},
                    {'name': 'Sofa', 'quantity': 2, 'description': 'Luxury sofas'},
                    {'name': 'Minibar', 'quantity': 1, 'description': 'Premium stocked minibar'},
                    {'name': 'Jacuzzi', 'quantity': 1, 'description': 'In-room jacuzzi'},
                    {'name': 'Safe', 'quantity': 1, 'description': 'Digital safe'}
                ]
            },
            {
                'number': '201',
                'type': 'Standard',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'Queen size bed'},
                    {'name': 'Chair', 'quantity': 2, 'description': 'Wooden chairs'},
                    {'name': 'TV', 'quantity': 1, 'description': '32-inch LCD TV'}
                ]
            },
            {
                'number': '202',
                'type': 'Deluxe',
                'inventory': [
                    {'name': 'Bed', 'quantity': 1, 'description': 'Queen size bed'},
                    {'name': 'Pillow', 'quantity': 1, 'description': 'Bed pillow'},
                    {'name': 'Nightstand', 'quantity': 2, 'description': 'Bedside tables'},
                    {'name': 'Lamp', 'quantity': 3, 'description': 'Room lamps'},
                    {'name': 'TV', 'quantity': 1, 'description': '42-inch Smart TV'},
                    {'name': 'Desk', 'quantity': 1, 'description': 'Work desk'},
                    {'name': 'Chair', 'quantity': 1, 'description': 'Desk chair'},
                    {'name': 'Artwork', 'quantity': 1, 'description': 'Wall art'},
                    {'name': 'Curtains', 'quantity': 1, 'description': 'Window curtains'},
                    {'name': 'Window', 'quantity': 1, 'description': 'Room window'},
                    {'name': 'Armchair', 'quantity': 1, 'description': 'Comfortable armchair'},
                    {'name': 'Ottoman', 'quantity': 1, 'description': 'Footrest'},
                    {'name': 'Table', 'quantity': 1, 'description': 'Coffee table'},
                    {'name': 'Headboard', 'quantity': 1, 'description': 'Bed headboard'},
                    {'name': 'Blanket', 'quantity': 1, 'description': 'Bed blanket'}
                ]
            }
        ]

        # Create the rooms and inventory
        rooms_created = 0
        inventory_items_created = 0

        for room in room_data:
            # Check if room already exists
            existing_room = Room.objects.filter(manager=manager, room_number=room['number']).first()
            
            if existing_room:
                self.stdout.write(f"Room {room['number']} already exists, skipping.")
                room_obj = existing_room
            else:
                # Create new room
                room_obj = Room.objects.create(
                    manager=manager,
                    room_number=room['number'],
                    room_type=room['type']
                )
                rooms_created += 1
                self.stdout.write(f"Created room {room['number']} - {room['type']}")
            
            # Add inventory items
            for item in room['inventory']:
                # Check if inventory item already exists
                existing_item = RoomInventory.objects.filter(
                    room=room_obj,
                    item_name=item['name']
                ).first()
                
                if existing_item:
                    self.stdout.write(f"  - {item['name']} already exists in room {room['number']}, updating quantity.")
                    existing_item.quantity = item['quantity']
                    existing_item.description = item['description']
                    existing_item.save()
                else:
                    # Create new inventory item
                    RoomInventory.objects.create(
                        room=room_obj,
                        item_name=item['name'],
                        quantity=item['quantity'],
                        description=item['description']
                    )
                    inventory_items_created += 1
                    self.stdout.write(f"  - Added {item['quantity']} {item['name']}(s) to room {room['number']}")
        
        self.stdout.write(self.style.SUCCESS(f'Successfully created {rooms_created} rooms and {inventory_items_created} inventory items!')) 