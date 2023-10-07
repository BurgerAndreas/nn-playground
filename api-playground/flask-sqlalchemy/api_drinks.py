from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///drinks.db'
db = SQLAlchemy(app)



# define object relational mapping
class Drink(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(80), unique=True, nullable=False)
  description = db.Column(db.String(120), unique=True, nullable=False)

  # define the string representation of the object
  def __repr__(self):
    # retturn f"Drink('{self.name}', '{self.description}')"
    return '<Drink %r>' % self.name

# default
@app.route('/', methods=['GET'])
def index():
  return 'Welcome to the drinks API', 200

# get all drinks
@app.route('/drinks', methods=['GET'])
def get_all_drinks():
  drinks = Drink.query.all()
  # instead of 
  # return {'drinks': drinks}
  # jsonify the result
  output = []
  for drink in drinks:
    drink_data = {}
    drink_data['name'] = drink.name
    drink_data['description'] = drink.description
    output.append(drink_data)
  return {'drinks': output}, 200


# get a single drink
@app.route('/drinks/<drink_id>', methods=['GET'])
def get_drink(drink_id):
  # drink = Drink.query.filter_by(id=drink_id).first()
  drink = Drink.query.get_or_404(drink_id) 
  # return jsonify({'drink': drink})
  return jsonify({'name': drink.name, 'description': drink.description}), 200


# add a new drink
@app.route('/drinks', methods=['POST'])
def add_drink():
  drink = Drink(name=request.json['name'], description=request.json['description'])
  db.session.add(drink)
  db.session.commit()
  # return jsonify({'drink': drink})
  return {'id': drink.id}, 201


# update a drink
@app.route('/drinks/<drink_id>', methods=['PUT'])
def update_drink(drink_id):
  drink = Drink.query.get_or_404(drink_id)
  drink.name = request.json['name']
  drink.description = request.json['description']
  db.session.commit()
  # return jsonify({'drink': drink})
  return {'id': drink.id}, 204


# delete a drink
@app.route('/drinks/<drink_id>', methods=['DELETE'])
def delete_drink(drink_id):
  drink = Drink.query.get_or_404(drink_id)
  db.session.delete(drink)
  db.session.commit()
  return {'id': drink.id}, 204