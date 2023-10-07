from api_drinks import db, Drink

# create the database and the database table
db.create_all()

# insert drink data
drink = Drink(name='Coke', description='A carbonated soft drink')
db.session.add(drink)
# or
db.session.add(Drink(name='Sprite', description='A lemon-lime flavored soft drink'))
# commit the changes
db.session.commit()

# query all drinks
Drink.query.all()